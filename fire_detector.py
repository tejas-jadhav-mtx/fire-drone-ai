"""
Fire detection + ball-launch mission logic - Raspberry Pi 4 (4GB) side
-----------------------------------------------------------------------
Capstone: RC-piloted firefighting quadcopter with autonomous target
detection and launch.

The AIRFRAME is flown by a human pilot over RC -- this script does not fly
the drone. What it owns is everything from "is there a fire in frame" to
"pull the trigger": optical+thermal fire detection, fusion, temporal
confirmation, and commanding the launcher, while the pilot handles
navigation (and sees the same video feed this script is analyzing).

ARCHITECTURE (why it's split this way):
  Raspberry Pi 4  -> perception + fusion + mission state machine + MAVLink.
                     Linux on the Pi is NOT a real-time OS: a camera read, a
                     frame of numpy math, or a garbage-collection pause can
                     stall this process for tens of milliseconds at a time.
  ESP8266         -> owns the launcher. It gets one dumb command ("FIRE")
                     over serial and handles the motor-burst timing itself,
                     on hardware that has nothing else to do. See the
                     companion esp8266_launcher.ino for that side.

This script is written so you can bench-test the detection+fusion+FSM logic
BEFORE your thermal camera, ESP8266, or Pixhawk are wired up: every hardware
dependency sits behind a small interface, and `--selftest` below exercises
the whole pipeline with synthetic data and fakes. Run that first, on just
your laptop, to confirm the state machine itself is sound.

SAFETY (read before you flip FULL_AUTO to True):
This launcher throws a 1.3-2 kg rigid object. The default in this script is
SEMI-AUTONOMOUS: the pipeline detects and confirms a fire on its own, but
will NOT fire without an explicit human confirmation (see ConfirmGate below
-- with RC flight, that confirmation is a dedicated switch on the pilot's
own transmitter by default, read back through the Pixhawk: see
RCSwitchConfirmGate). Flipping FULL_AUTO to True removes that human step
entirely -- treat that as a safety-case decision, not a config toggle.
"""

from __future__ import annotations

import csv
import logging
import os
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # optical detector will refuse to run without it; selftest doesn't need it

try:  # the impact gate and tracker live in their own files (same folder)
    from impact_predictor import CameraModel, ImpactGate, predict_from_candidate, from_mavlink, pilot_cue
    from target_tracker import TargetTracker
except ImportError:
    CameraModel = ImpactGate = predict_from_candidate = from_mavlink = pilot_cue = TargetTracker = None


# ===========================================================================
# CONFIG - the knobs you'll actually touch. Nothing else below should need
# editing to bring this up on the bench.
# ===========================================================================

FULL_AUTO = False                  # see SAFETY note above. Default: human must confirm.
ENABLE_THERMAL = False              # camera-only by default (budget build); set True only if an MLX90640 is fitted
ENABLE_LAUNCHER = True
ENABLE_MAVLINK = True               # RC flight -> the confirm switch and the altitude gate both read through the Pixhawk link

CAMERA_INDEX = 0                   # /dev/video0 -- change if using picamera2 instead (see note in CV2Camera)
THERMAL_I2C_ADDR = 0x33            # MLX90640 default address

# Fire-color heuristic thresholds (HSV, OpenCV convention: H in 0-179)
FIRE_HUE_LOW, FIRE_HUE_HIGH = 0, 45     # red through orange/yellow
FIRE_SAT_MIN = 120
FIRE_VAL_MIN = 150
FIRE_MIN_AREA_PX = 150                  # ignore specks smaller than this (lens flare, embers at distance)

# Thermal hotspot threshold. MLX90640 breakouts are commonly spec'd to a
# ~300C usable ceiling -- real flame is hotter than that, so pixels will
# often read "pegged at max" rather than a clean absolute temperature.
# Treat near-saturation the same as a confirmed hotspot; don't rely on the
# absolute number alone. VERIFY your specific breakout's datasheet range
# before trusting this constant.
THERMAL_HOTSPOT_C = 100.0
THERMAL_SATURATION_MARGIN_C = 15.0
THERMAL_MIN_CLUSTER_PX = 3               # MLX90640 is only 32x24 -- a real fire is rarely just 1 pixel

# Optical<->thermal fusion: both sensors are assumed front-facing and
# roughly co-located (not true geometric calibration -- see note in
# fuse_detections()). This is the normalized-coordinate tolerance for
# treating an optical and a thermal candidate as "the same object".
FUSION_TOLERANCE = 0.28

# Temporal confirmation: require fused detections in at least this many of
# the last WINDOW frames before treating it as a real, not noise.
CONFIRM_WINDOW = 10
CONFIRM_COUNT = 7
CONFIRM_RELEASE_COUNT = 3          # once confirmed, stay confirmed until this few (or fewer) of the last WINDOW results
                                   # still show fire. Stops flip-flopping when the detector's confidence hovers near the threshold.

# Impact gate: the launcher may only be armed when the predicted landing point
# of the ball is inside the fire (see impact_predictor.py). Needs altitude and
# velocity from the flight controller; with none available it fails closed.
ENABLE_IMPACT_GATE = True
DETECT_EVERY_N = 1                 # run the detector on every Nth frame; in-between frames reuse the last result
IMPACT_COAST_STEPS = 2             # a target the detector loses may be used for this many detector runs (position
                                   # extrapolated by the tracker) before the impact gate drops it. Fails closed after.

COOLDOWN_S = 5.0                  # after a shot, ignore new detections this long

# How the pilot confirms a fire before the launcher is allowed to fire.
# "rc_switch" (default): a dedicated 2-position switch on the pilot's own
#   transmitter, read back as a channel value through the Pixhawk -- no new
#   hardware, the pilot already has their hands on the one input device
#   that matters. Requires ENABLE_MAVLINK=True and the channel/threshold
#   below to match your actual transmitter/receiver setup.
# "file": the original ground-station placeholder (touch a file to
#   confirm). Kept only for bench-testing the pipeline from a laptop when
#   nothing is bound to a transmitter yet.
CONFIRM_MODE = "rc_switch"

CONFIRM_RC_CHANNEL = 7                  # 1-indexed; pick a spare channel on your transmitter and set this to match
CONFIRM_RC_THRESHOLD_US = 1700          # PWM microseconds above which the switch counts as "confirmed" -- check your radio's endpoints in Mission Planner first, this is a typical default, not a measured one

CONFIRM_FLAG_PATH = "/home/pi/confirm_fire.flag"   # only used when CONFIRM_MODE == "file"
CONFIRM_FLAG_MAX_AGE_S = 8.0       # a stale flag file doesn't count as a fresh confirm

LAUNCHER_SERIAL_PORT = "/dev/ttyUSB0"   # ESP8266 over USB (simplest -- see wiring note in chat)
LAUNCHER_BAUD = 115200
MAGAZINE_CAPACITY = 3

LOG_CSV_PATH = "mission_log.csv"

# If MAVLink is enabled, optionally gate firing to a sane altitude band
# above the actual drop point (ties back to the TF-Luna rangefinder this
# project already carries for exactly this reason).
ALT_GATE_ENABLED = True
ALT_MIN_M, ALT_MAX_M = 2.0, 8.0


# ===========================================================================
# Data types
# ===========================================================================

@dataclass
class Candidate:
    cx: float   # normalized 0-1 within that sensor's own frame
    cy: float
    area: float
    strength: float  # meaning differs by sensor (pixel count vs peak temp); comparable within-sensor only


class MissionState(Enum):
    SEARCH = auto()
    CANDIDATE = auto()
    AWAITING_CONFIRM = auto()
    FIRING = auto()
    COOLDOWN = auto()
    MAGAZINE_EMPTY = auto()


# ===========================================================================
# Sensor interfaces + real implementations
# (each has a Fake* counterpart further down, used by --selftest)
# ===========================================================================

class OpticalSource(ABC):
    @abstractmethod
    def read_frame(self) -> Optional[np.ndarray]:
        """Returns a BGR uint8 frame, or None if a read failed."""


class ThermalSource(ABC):
    @abstractmethod
    def read_frame(self) -> Optional[np.ndarray]:
        """Returns a (24, 32) float32 array in degrees C, or None if a read failed."""


class LauncherLink(ABC):
    @abstractmethod
    def send(self, cmd: str) -> str:
        """Sends one command line, returns the first response line (stripped)."""

    def fire_one(self) -> str:
        """ARM then FIRE, per the serial protocol the ESP8266 firmware expects.
        Returns the final result token: FIRED:<n> / JAM / EMPTY / NOTARMED."""
        armed = self.send("ARM")
        if armed != "ARMED":
            return f"ARM_FAILED:{armed}"
        self.send("FIRE")  # first reply is "FIRING", echo/ack -- read and discard
        return self.send("")  # second reply is the actual result line


class ConfirmGate(ABC):
    @abstractmethod
    def is_confirmed(self) -> bool:
        """Human-in-the-loop check. Default implementation reads a switch
        on the RC pilot's own transmitter (RCSwitchConfirmGate); FileConfirmGate
        is kept only for bench-testing without a bound transmitter."""


class MavlinkTelemetry(ABC):
    @abstractmethod
    def get_relative_alt_m(self) -> Optional[float]:
        ...

    @abstractmethod
    def get_rc_channel_us(self, channel: int) -> Optional[int]:
        """Returns the given 1-indexed RC channel's raw value in
        microseconds (typically ~1000-2000us), or None if no RC_CHANNELS
        message has been seen yet."""

    def get_flight_state(self):
        """(altitude_m, v_forward, v_right, roll_deg, pitch_deg) or None if unknown."""
        return None


class FixedFlightState:
    """BENCH ONLY: pretends the drone hovers/moves with fixed numbers so the
    impact gate can be exercised on a laptop with a replay video. These are
    made-up values, not measurements."""
    def __init__(self, alt_m=5.0, v_fwd=0.0, v_right=0.0, roll_deg=0.0, pitch_deg=0.0):
        self.state = (alt_m, v_fwd, v_right, roll_deg, pitch_deg)

    def get_flight_state(self):
        return self.state


# ---- Real implementations ------------------------------------------------

class CV2Camera(OpticalSource):
    def __init__(self, index: int = CAMERA_INDEX):
        if cv2 is None:
            raise RuntimeError("opencv-python is not installed (pip install opencv-python)")
        # NOTE: if your "CAMERA" BOM line is the Pi CSI ribbon camera rather
        # than a USB webcam, swap this for picamera2's Picamera2() class --
        # cv2.VideoCapture does not talk to the CSI port on a Pi 4.
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {index}")

    def read_frame(self) -> Optional[np.ndarray]:
        ok, frame = self.cap.read()
        return frame if ok else None


class PiCamera2Source(OpticalSource):
    """Raspberry Pi CSI camera (Camera Module 2/3) through picamera2.
    cv2.VideoCapture does not read the CSI port on a Pi 4 with the current
    camera stack. Install: sudo apt install python3-picamera2
    UNTESTED on real hardware -- written from the picamera2 API, check on the Pi."""
    def __init__(self, size=(1280, 720)):
        from picamera2 import Picamera2  # imported here so this file loads without it
        self.cam = Picamera2()
        self.cam.configure(self.cam.create_video_configuration(main={"size": size, "format": "RGB888"}))
        self.cam.start()

    def read_frame(self) -> Optional[np.ndarray]:
        frame = self.cam.capture_array()          # "RGB888" arrays are BGR-ordered for OpenCV
        return frame if frame is not None else None


class VideoFileSource(OpticalSource):
    """Replay a recorded video through the full pipeline (bench / demo-backup mode).
    Returns None at the end of the file unless loop=True."""
    def __init__(self, path: str, loop: bool = False):
        if cv2 is None:
            raise RuntimeError("opencv-python is not installed")
        self.path, self.loop = path, loop
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video file {path!r}")
        self.finished = False

    def read_frame(self) -> Optional[np.ndarray]:
        ok, frame = self.cap.read()
        if not ok and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        if not ok:
            self.finished = True
            return None
        return frame


class MLX90640Thermal(ThermalSource):
    def __init__(self, i2c_addr: int = THERMAL_I2C_ADDR):
        # Deps imported here, not at module level, so this file still loads
        # (and --selftest still runs) on a machine without the sensor libs.
        import board
        import busio
        import adafruit_mlx90640
        i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
        self.sensor = adafruit_mlx90640.MLX90640(i2c)
        self.sensor.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ
        self._buf = [0.0] * 768

    def read_frame(self) -> Optional[np.ndarray]:
        try:
            self.sensor.getFrame(self._buf)
        except ValueError:
            return None  # MLX90640 occasionally drops a frame over I2C; caller just retries next loop
        return np.array(self._buf, dtype=np.float32).reshape((24, 32))


class SerialLauncherLink(LauncherLink):
    def __init__(self, port: str = LAUNCHER_SERIAL_PORT, baud: int = LAUNCHER_BAUD):
        import serial  # pyserial
        self.ser = serial.Serial(port, baud, timeout=2)
        time.sleep(2)  # ESP8266 resets on serial-port-open; give its setup() time to finish

    def send(self, cmd: str) -> str:
        if cmd:
            self.ser.write((cmd + "\n").encode("ascii"))
        line = self.ser.readline().decode("ascii", errors="replace").strip()
        return line


class FileConfirmGate(ConfirmGate):
    """Bench-test placeholder -- touch CONFIRM_FLAG_PATH to approve a fire.
    Useful when testing the pipeline on a laptop with no transmitter bound.
    See RCSwitchConfirmGate for the one actually meant for flight."""
    def is_confirmed(self) -> bool:
        try:
            age = time.time() - os.path.getmtime(CONFIRM_FLAG_PATH)
        except FileNotFoundError:
            return False
        if age <= CONFIRM_FLAG_MAX_AGE_S:
            os.remove(CONFIRM_FLAG_PATH)  # consume it -- one confirm fires at most one shot
            return True
        return False


class RCSwitchConfirmGate(ConfirmGate):
    """Reads a dedicated switch on the pilot's own RC transmitter, passed
    through by the Pixhawk and read here over MAVLink -- no extra hardware,
    and the pilot already has their hands on the one input device that
    matters while flying. Requires a live MavlinkTelemetry connection.

    Before this means anything: in Mission Planner (or your GCS of choice),
    go to the radio calibration screen, flip your chosen physical switch,
    and confirm (a) which CONFIRM_RC_CHANNEL number it actually shows up
    as, and (b) what PWM value it reads in each position -- then set
    CONFIRM_RC_CHANNEL / CONFIRM_RC_THRESHOLD_US to match. The defaults
    here are typical, not measured on your radio.
    """
    def __init__(self, telemetry: MavlinkTelemetry,
                 channel: int = CONFIRM_RC_CHANNEL, threshold_us: int = CONFIRM_RC_THRESHOLD_US):
        self.telemetry = telemetry
        self.channel = channel
        self.threshold_us = threshold_us

    def is_confirmed(self) -> bool:
        us = self.telemetry.get_rc_channel_us(self.channel)
        if us is None:
            return False  # no RC_CHANNELS message seen yet -- fail closed, never fail open on "unknown"
        return us >= self.threshold_us


class PymavlinkTelemetry(MavlinkTelemetry):
    def __init__(self, device: str = "/dev/ttyACM0", baud: int = 57600):
        from pymavlink import mavutil
        self.conn = mavutil.mavlink_connection(device, baud=baud)
        self._last_alt = None
        self._last_rc: dict[int, int] = {}
        self._gpos = None
        self._att = None
        self._state_time = 0.0

    def _pump(self):
        """Drain every pending message and keep the newest of each type we use.
        (recv_match(type=X) throws away other types it skips over, so reading
        three message types with three filtered calls could lose data.)
        UNTESTED against a real Pixhawk."""
        while True:
            msg = self.conn.recv_match(blocking=False)
            if msg is None:
                return
            t = msg.get_type()
            if t == "GLOBAL_POSITION_INT":
                self._gpos = msg
                self._last_alt = msg.relative_alt / 1000.0  # mm -> m
                self._state_time = time.monotonic()
            elif t == "ATTITUDE":
                self._att = msg
            elif t == "RC_CHANNELS":
                for i in range(1, 19):  # RC_CHANNELS carries chan1_raw..chan18_raw
                    field = f"chan{i}_raw"
                    if hasattr(msg, field):
                        self._last_rc[i] = getattr(msg, field)

    def get_relative_alt_m(self) -> Optional[float]:
        self._pump()
        return self._last_alt

    def get_rc_channel_us(self, channel: int) -> Optional[int]:
        self._pump()
        return self._last_rc.get(channel)

    def get_flight_state(self):
        self._pump()
        if self._gpos is None or self._att is None or time.monotonic() - self._state_time > 1.0:
            return None   # missing or older than 1 s: the impact gate then fails closed
        return from_mavlink(self._gpos, self._att)


# ===========================================================================
# Detection logic
# ===========================================================================

def detect_fire_color(frame_bgr: np.ndarray) -> list[Candidate]:
    """Classical HSV flame-color heuristic. No training data needed, runs
    comfortably real-time on a Pi 4. Swap this out for a trained detector
    (e.g. a YOLOv8n flame/smoke model) later without touching anything
    else -- it just needs to keep returning the same Candidate list."""
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        (FIRE_HUE_LOW, FIRE_SAT_MIN, FIRE_VAL_MIN),
        (FIRE_HUE_HIGH, 255, 255),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < FIRE_MIN_AREA_PX:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = (M["m10"] / M["m00"]) / w
        cy = (M["m01"] / M["m00"]) / h
        out.append(Candidate(cx=cx, cy=cy, area=area, strength=area))
    return out


def detect_thermal_hotspots(thermal_frame: np.ndarray) -> list[Candidate]:
    """Finds clusters of pixels at or near the sensor's hot ceiling.
    thermal_frame: (24, 32) float32, degrees C."""
    h, w = thermal_frame.shape
    peak = float(np.max(thermal_frame))
    mask = (thermal_frame >= THERMAL_HOTSPOT_C) | (thermal_frame >= peak - THERMAL_SATURATION_MARGIN_C)
    mask_u8 = (mask.astype(np.uint8)) * 255

    # Connected components, not just a global centroid -- two separate hot
    # objects (e.g. a car engine AND a real fire) shouldn't get averaged
    # into one meaningless point.
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    out = []
    for label in range(1, n_labels):  # label 0 is background
        pixel_count = stats[label, cv2.CC_STAT_AREA]
        if pixel_count < THERMAL_MIN_CLUSTER_PX:
            continue
        cx, cy = centroids[label]
        cluster_peak = float(np.max(thermal_frame[labels == label]))
        out.append(Candidate(cx=cx / w, cy=cy / h, area=pixel_count, strength=cluster_peak))
    return out


def fuse_detections(optical: list[Candidate], thermal: list[Candidate]) -> list[Candidate]:
    """Pairs an optical flame-color blob with a thermal hotspot at roughly
    the same normalized position.

    HONEST LIMITATION: this assumes the two cameras are front-facing,
    co-located, and share a boresight closely enough that normalized
    centroids line up within FUSION_TOLERANCE. That is an approximation,
    not a calibration. If you want this tolerance number to mean something
    precise, do a real intrinsic/extrinsic calibration (e.g. a checkerboard
    visible to both the optical and thermal sensor) and replace this
    distance check with a proper homography. For a bench-test capstone
    demo at short range, the approximation is usually good enough -- just
    don't present FUSION_TOLERANCE as a calibrated number in your report.
    """
    fused = []
    for o in optical:
        best = None
        best_d = FUSION_TOLERANCE
        for t in thermal:
            d = ((o.cx - t.cx) ** 2 + (o.cy - t.cy) ** 2) ** 0.5
            if d <= best_d:
                best_d = d
                best = t
        if best is not None:
            fused.append(Candidate(
                cx=(o.cx + best.cx) / 2, cy=(o.cy + best.cy) / 2,
                area=o.area, strength=best.strength,
            ))
    return fused


# ===========================================================================
# Mission state machine
# ===========================================================================

class MissionFSM:
    def __init__(self, optical: OpticalSource, thermal: Optional[ThermalSource],
                 launcher: Optional[LauncherLink], confirm_gate: ConfirmGate,
                 telemetry: Optional[MavlinkTelemetry] = None, logger: Optional[logging.Logger] = None,
                 detector=None, flight=None, camera=None, every_n: int = DETECT_EVERY_N):
        self.detector = detector or detect_fire_color   # swap in the YOLO detector with --detector yolo
        self.every_n = max(1, int(every_n))
        self._frame_i = 0
        self.det_calls = 0
        self.det_seconds = 0.0
        self._t_start = time.time()
        self._last_optical: list[Candidate] = []
        self.flight = flight                       # anything with get_flight_state(); None = impact gate off
        self.camera = camera
        self.use_impact = bool(flight is not None and ImpactGate is not None and ENABLE_IMPACT_GATE)
        self.impact_gate = ImpactGate(base_gate=None) if self.use_impact else None
        self.tracker = TargetTracker() if self.use_impact else None
        self._track_t = None
        self.last_pred = None
        self._last_hold_log = 0.0
        self.optical = optical
        self.thermal = thermal
        self.launcher = launcher
        self.confirm_gate = confirm_gate
        self.telemetry = telemetry
        self.log = logger or logging.getLogger("mission")

        self.state = MissionState.SEARCH
        self.confirm_history: deque[bool] = deque(maxlen=CONFIRM_WINDOW)
        self.balls_remaining = MAGAZINE_CAPACITY
        self.cooldown_until = 0.0
        self.last_fused: list[Candidate] = []
        self._confirmed = False

        self._csv = open(LOG_CSV_PATH, "a", newline="")
        self._csv_writer = csv.writer(self._csv)
        if self._csv.tell() == 0:
            self._csv_writer.writerow(["t", "state", "n_optical", "n_thermal", "n_fused", "alt_m", "event"])

    def _alt_ok(self) -> bool:
        if not (ALT_GATE_ENABLED and self.telemetry is not None):
            return True  # no telemetry wired up -- don't block the bench-test FSM on it
        alt = self.telemetry.get_relative_alt_m()
        if alt is None:
            return False  # haven't heard a position fix yet -- don't fire blind
        return ALT_MIN_M <= alt <= ALT_MAX_M

    def _log_row(self, n_opt: int, n_therm: int, n_fused: int, event: str = ""):
        alt = self.telemetry.get_relative_alt_m() if self.telemetry else None
        self._csv_writer.writerow([time.time(), self.state.name, n_opt, n_therm, n_fused, alt, event])
        self._csv.flush()

    def step(self) -> MissionState:
        now = time.time()

        if self.state == MissionState.COOLDOWN:
            if now >= self.cooldown_until:
                self.state = MissionState.SEARCH if self.balls_remaining > 0 else MissionState.MAGAZINE_EMPTY
            return self.state

        if self.state == MissionState.MAGAZINE_EMPTY:
            return self.state  # terminal for this mission leg; see main() for RTL handling

        frame = self.optical.read_frame()
        self._frame_i += 1
        ran_detector = False
        if frame is None:
            optical_c = []
        elif (self._frame_i - 1) % self.every_n == 0:
            _t0 = time.perf_counter()
            optical_c = self.detector(frame)
            self.det_seconds += time.perf_counter() - _t0
            self.det_calls += 1
            self._last_optical = optical_c
            ran_detector = True
        else:
            optical_c = self._last_optical   # in-between frame: reuse the last detection

        thermal_c: list[Candidate] = []
        if self.thermal is not None:
            tframe = self.thermal.read_frame()
            if tframe is not None:
                thermal_c = detect_thermal_hotspots(tframe)

        fused = fuse_detections(optical_c, thermal_c) if self.thermal is not None else [
            Candidate(c.cx, c.cy, c.area, c.strength) for c in optical_c
        ]
        self.last_fused = fused
        # The 7-of-10 rule counts DETECTOR RESULTS, not camera frames: frames that
        # only reuse the last result would otherwise count the same detection twice.
        if ran_detector or frame is None:
            self.confirm_history.append(len(fused) > 0)
        if self.use_impact:
            self._update_impact(fused, ran_detector)
        self._log_row(len(optical_c), len(thermal_c), len(fused))

        hits = sum(self.confirm_history)
        if hits >= CONFIRM_COUNT:
            self._confirmed = True
        elif hits <= CONFIRM_RELEASE_COUNT:
            self._confirmed = False          # between the two limits: keep the previous answer (hysteresis)
        confirmed = self._confirmed

        if self.state == MissionState.SEARCH:
            if confirmed:
                self.state = MissionState.AWAITING_CONFIRM
                self.log.info("Fire confirmed across %d/%d recent frames -> AWAITING_CONFIRM",
                              sum(self.confirm_history), len(self.confirm_history))
                self._log_row(len(optical_c), len(thermal_c), len(fused), "fire_confirmed")

        elif self.state == MissionState.AWAITING_CONFIRM:
            if not confirmed:
                self.state = MissionState.SEARCH  # it was noise, drop back
                return self.state
            if not self._alt_ok():
                self.log.warning("Fire confirmed but altitude gate not satisfied -- holding")
                return self.state
            if self.use_impact and not self.impact_gate.is_confirmed():
                # Checked BEFORE the pilot switch, so a file/switch confirm is
                # never used up on a shot that would miss.
                if time.time() - self._last_hold_log > 1.0:
                    self._last_hold_log = time.time()
                    p = self.last_pred
                    self.log.info("Holding: %s%s", p.reason if p else "no usable target/flight data",
                                  f" -> {pilot_cue(p)}" if p else "")
                return self.state
            ready = FULL_AUTO or self.confirm_gate.is_confirmed()
            if ready:
                self.state = MissionState.FIRING

        if self.state == MissionState.FIRING:
            self._do_fire()

        return self.state

    def timing_summary(self) -> str:
        wall = max(time.time() - self._t_start, 1e-9)
        per = (1000.0 * self.det_seconds / self.det_calls) if self.det_calls else 0.0
        return (f"{self._frame_i} frames read, detector ran {self.det_calls} times, "
                f"mean {per:.1f} ms per detector call, {self._frame_i / wall:.1f} frames/s overall "
                f"(includes the cooldown wait and the 50 ms loop sleep)")

    def _update_impact(self, fused, ran_detector: bool):
        """Keep a smoothed target track and refresh the impact prediction."""
        now = time.monotonic()
        if ran_detector:
            dt = 0.1 if self._track_t is None else min(max(now - self._track_t, 0.02), 1.0)
            self._track_t = now
            self.tracker.step(fused, dt)
        live = [t for t in self.tracker.tracks if t.misses <= IMPACT_COAST_STEPS]
        fs = self.flight.get_flight_state()
        if not live or fs is None:
            self.last_pred = None
            self.impact_gate.update(None)
            return
        t = max(live, key=lambda tr: sum(tr.hits))
        alt, vf, vr, roll, pitch = fs
        pred = predict_from_candidate(t.pos[0], t.pos[1], alt, vf, vr, self.camera or CameraModel(),
                                      roll_deg=roll, pitch_deg=pitch)
        self.last_pred = pred
        self.impact_gate.update(pred)

    def _do_fire(self):
        if self.launcher is None:
            self.log.warning("FIRING state reached but no launcher link configured -- skipping actual fire")
            self.state = MissionState.COOLDOWN
            self.cooldown_until = time.time() + COOLDOWN_S
            return

        result = self.launcher.fire_one()
        self.log.info("Launcher result: %s", result)
        self._log_row(0, 0, 0, f"fire_result:{result}")

        if result.startswith("FIRED:"):
            self.balls_remaining = int(result.split(":")[1])
        elif result == "JAM":
            self.log.error("Launcher reported JAM -- ball may not have left the barrel. Inspect before next shot.")
        elif result == "EMPTY":
            self.balls_remaining = 0

        if self.last_pred is not None:
            self._log_row(0, 0, 0, f"predicted_miss_m:{self.last_pred.miss_m:.2f}")
        self.confirm_history.clear()
        self._confirmed = False
        if self.use_impact:
            self.tracker = TargetTracker()      # a new target after the shot
            self.impact_gate.update(None)
            self._track_t = None
        self.cooldown_until = time.time() + COOLDOWN_S
        self.state = MissionState.COOLDOWN


# ===========================================================================
# Fakes for --selftest (no hardware required)
# ===========================================================================

class FakeOpticalSource(OpticalSource):
    """Yields a blank frame, then a frame with a bright orange blob, forever after."""
    def __init__(self, fire_after_frame: int = 3):
        self.i = 0
        self.fire_after_frame = fire_after_frame

    def read_frame(self) -> Optional[np.ndarray]:
        self.i += 1
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        if self.i >= self.fire_after_frame:
            cv2.circle(frame, (160, 120), 25, (20, 90, 220), -1)  # BGR: strong red/orange
        return frame


class FakeThermalSource(ThermalSource):
    def __init__(self, fire_after_frame: int = 3):
        self.i = 0
        self.fire_after_frame = fire_after_frame

    def read_frame(self) -> Optional[np.ndarray]:
        self.i += 1
        frame = np.full((24, 32), 22.0, dtype=np.float32)  # ambient
        if self.i >= self.fire_after_frame:
            frame[10:14, 14:18] = 280.0  # hot cluster roughly centered, matching the optical blob
        return frame


class FakeLauncherLink(LauncherLink):
    def __init__(self):
        self.balls = MAGAZINE_CAPACITY
        self.armed = False

    def send(self, cmd: str) -> str:
        if cmd == "ARM":
            self.armed = True
            return "ARMED"
        if cmd == "":  # stand-in for "read the result of a FIRE already sent"
            if not self.armed:
                return "NOTARMED"
            self.armed = False
            if self.balls <= 0:
                return "EMPTY"
            self.balls -= 1
            return f"FIRED:{self.balls}"
        return "OK"


class AlwaysConfirmGate(ConfirmGate):
    """For the selftest: approve the first time it's asked, so the FSM
    actually reaches FIRING instead of waiting on a human forever."""
    def __init__(self):
        self._asked = False

    def is_confirmed(self) -> bool:
        if not self._asked:
            self._asked = True
            return True
        return False


def run_selftest():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("selftest")
    fsm = MissionFSM(
        optical=FakeOpticalSource(fire_after_frame=3),
        thermal=FakeThermalSource(fire_after_frame=3),
        launcher=FakeLauncherLink(),
        confirm_gate=AlwaysConfirmGate(),
        telemetry=None,
        logger=log,
    )
    states_seen = []
    for frame_i in range(CONFIRM_WINDOW + 10):
        s = fsm.step()
        states_seen.append(s)
        if s == MissionState.MAGAZINE_EMPTY:
            break

    assert MissionState.AWAITING_CONFIRM in states_seen, "never reached AWAITING_CONFIRM -- fusion/confirm logic broken"
    assert MissionState.FIRING in states_seen or MissionState.COOLDOWN in states_seen, "never reached FIRING -- confirm gate wiring broken"
    assert fsm.balls_remaining == MAGAZINE_CAPACITY - 1, f"expected {MAGAZINE_CAPACITY - 1} balls left, got {fsm.balls_remaining}"
    log.info("SELFTEST PASSED: states=%s, balls_remaining=%d", [s.name for s in states_seen], fsm.balls_remaining)
    os.remove(LOG_CSV_PATH) if os.path.exists(LOG_CSV_PATH) else None
    run_impact_selftest(log)


def run_impact_selftest(log):
    """Impact gate + detect-every-N wiring, on fake data."""
    if ImpactGate is None:
        log.info("impact_predictor.py / target_tracker.py not found: impact selftest skipped")
        return
    down = CameraModel(62.2, 48.8, 90.0)     # looking straight down, so the blob at the image centre is right below

    def run(alt, vfwd, n_steps=CONFIRM_WINDOW + 12, every_n=1, calls=None):
        def det(f):
            if calls is not None:
                calls.append(1)
            return detect_fire_color(f)
        fsm = MissionFSM(FakeOpticalSource(fire_after_frame=1), None, FakeLauncherLink(), AlwaysConfirmGate(),
                         None, log, detector=det, flight=FixedFlightState(alt_m=alt, v_fwd=vfwd),
                         camera=down, every_n=every_n)
        for _ in range(n_steps):
            fsm.step()
        return fsm

    f = run(alt=5.0, vfwd=0.0)
    assert f.balls_remaining == MAGAZINE_CAPACITY - 1, "hovering over the fire: should have fired"
    f = run(alt=5.0, vfwd=3.0)
    assert f.balls_remaining == MAGAZINE_CAPACITY, "moving at 3 m/s: predicted miss, must NOT fire"
    f = run(alt=1.0, vfwd=0.0)
    assert f.balls_remaining == MAGAZINE_CAPACITY, "1 m altitude is outside the 2-8 m band: must NOT fire"
    calls = []
    run(alt=5.0, vfwd=3.0, n_steps=20, every_n=2, calls=calls)     # moving, so no shot and no cooldown cuts the count short
    assert len(calls) == 10, f"every_n=2 should run the detector on 10 of 20 frames, ran {len(calls)}"
    f = run(alt=5.0, vfwd=0.0, n_steps=20, every_n=2)
    assert f.balls_remaining == MAGAZINE_CAPACITY - 1, "every_n=2 should still confirm and fire"
    log.info("IMPACT SELFTEST PASSED (hover fires, 3 m/s blocked, 1 m blocked, every-2nd-frame ok)")
    os.remove(LOG_CSV_PATH) if os.path.exists(LOG_CSV_PATH) else None


def parse_args(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Fire detection + ball-launch mission logic")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--source", choices=["picamera", "usb", "video"], default="picamera",
                    help="picamera = Pi CSI camera (flight default), usb = webcam, video = replay a file")
    ap.add_argument("--video", help="path to a recorded video (with --source video, or implied by it)")
    ap.add_argument("--loop", action="store_true", help="loop the replay video")
    ap.add_argument("--detector", choices=["color", "yolo"], default="color")
    ap.add_argument("--weights", help="YOLO weights file (.pt or ncnn folder) for --detector yolo")
    ap.add_argument("--imgsz", type=int, default=416, help="YOLO input size (416 = trained size; 320 is faster, check recall)")
    ap.add_argument("--every-n", type=int, default=DETECT_EVERY_N, help="run the detector on every Nth frame only")
    ap.add_argument("--no-impact", action="store_true", help="turn the impact gate off (old behaviour: pilot switch only)")
    ap.add_argument("--bench-alt", type=float, default=5.0, help="bench only: pretend altitude in metres")
    ap.add_argument("--bench-speed", type=float, default=0.0, help="bench only: pretend forward speed in m/s")
    ap.add_argument("--cam-hfov", type=float, default=62.2, help="camera horizontal FOV, degrees (PLACEHOLDER default)")
    ap.add_argument("--cam-vfov", type=float, default=48.8, help="camera vertical FOV, degrees (PLACEHOLDER default)")
    ap.add_argument("--cam-pitch", type=float, default=45.0, help="camera angle below horizon, degrees (PLACEHOLDER default; 90 = straight down)")
    ap.add_argument("--bench", action="store_true",
                    help="no Pixhawk, no real launcher: telemetry off, launcher simulated, confirm by file "
                         "(touch %s). Safe to run on a laptop." % CONFIRM_FLAG_PATH)
    return ap.parse_args(argv)


def main(args=None):
    args = args or parse_args([])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("mission")

    global CONFIRM_MODE
    enable_mavlink, enable_launcher = ENABLE_MAVLINK, ENABLE_LAUNCHER
    if args.bench:
        enable_mavlink, enable_launcher = False, False
        CONFIRM_MODE = "file"
        log.info("BENCH MODE: no Pixhawk, simulated launcher, confirm by touching %s", CONFIRM_FLAG_PATH)

    if args.video and args.source != "video":
        args.source = "video"
    if args.source == "video":
        if not args.video:
            raise SystemExit("--source video needs --video <path>")
        optical: OpticalSource = VideoFileSource(args.video, loop=args.loop)
    elif args.source == "usb":
        optical = CV2Camera()
    else:
        optical = PiCamera2Source()

    detector = None
    if args.detector == "yolo":
        if not args.weights:
            raise SystemExit("--detector yolo needs --weights <path>")
        from yolo_fire_detector import YoloFireModel, detect_fire_yolo
        _model = YoloFireModel(args.weights, imgsz=args.imgsz)
        detector = lambda f: detect_fire_yolo(f, _model)

    thermal: Optional[ThermalSource] = MLX90640Thermal() if ENABLE_THERMAL else None
    launcher: Optional[LauncherLink] = SerialLauncherLink() if enable_launcher else (FakeLauncherLink() if args.bench else None)
    telemetry: Optional[MavlinkTelemetry] = PymavlinkTelemetry() if enable_mavlink else None

    confirm_gate: ConfirmGate
    if CONFIRM_MODE == "rc_switch":
        if telemetry is None:
            raise RuntimeError(
                "CONFIRM_MODE is 'rc_switch' but ENABLE_MAVLINK is False, so there's no "
                "Pixhawk link to read the switch from. Either set ENABLE_MAVLINK=True, or "
                "set CONFIRM_MODE='file' for a bench test with no transmitter bound."
            )
        confirm_gate = RCSwitchConfirmGate(telemetry)
    elif CONFIRM_MODE == "file":
        confirm_gate = FileConfirmGate()
    else:
        raise ValueError(f"Unknown CONFIRM_MODE: {CONFIRM_MODE!r}")

    flight = None
    camera = None
    if not args.no_impact and ENABLE_IMPACT_GATE and CameraModel is not None:
        camera = CameraModel(args.cam_hfov, args.cam_vfov, args.cam_pitch)
        if args.bench:
            flight = FixedFlightState(alt_m=args.bench_alt, v_fwd=args.bench_speed)
            log.warning("BENCH: impact gate uses PRETEND flight data: altitude %.1f m, forward speed %.1f m/s",
                        args.bench_alt, args.bench_speed)
        else:
            flight = telemetry
    if flight is None:
        log.warning("Impact gate is OFF: the launcher can arm on fire confirmation + pilot switch alone.")

    fsm = MissionFSM(optical, thermal, launcher, confirm_gate, telemetry, log, detector=detector,
                     flight=flight, camera=camera, every_n=args.every_n)
    log.info("Mission FSM running. FULL_AUTO=%s, CONFIRM_MODE=%s. Ctrl+C to stop.", FULL_AUTO, CONFIRM_MODE)

    try:
        while True:
            state = fsm.step()
            if getattr(optical, 'finished', False):
                log.info("Replay video finished.")
                break
            if state == MissionState.MAGAZINE_EMPTY:
                log.warning("Magazine empty. Recommend RTL.")
                if ENABLE_MAVLINK:
                    pass  # deliberately not auto-switching flight mode here -- see AUTO_RTL note in chat
                break
            time.sleep(0.05)  # ~20 Hz; MLX90640 itself is the real bottleneck at 2-8 Hz
    except KeyboardInterrupt:
        log.info("Stopped by operator.")
    log.info("TIMING: %s", fsm.timing_summary())


if __name__ == "__main__":
    _args = parse_args()
    if _args.selftest:
        run_selftest()
    else:
        main(_args)
