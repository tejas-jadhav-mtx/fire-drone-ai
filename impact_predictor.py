"""
Impact-point predictor and firing gate
----------------------------------------------------------------------------
IDEA: instead of only asking the pilot "do you confirm?", the Pi works out
WHERE THE BALL WILL LAND if it were released right now, and only treats the
shot as good when that point is inside the fire. It also tells the pilot
which way to nudge the drone.

WHY IT MATTERS: launcher_physics.py showed that with a fixed straight-down
launcher, accuracy is all about hover precision (drift limits of 0.87 / 0.45
/ 0.32 m/s at 2 / 5 / 8 m for a 0.3 m miss). This module turns that fixed
table into a live decision using the drone's actual speed and the fire's
actual position in the camera image.

WHAT IS IN HERE
  1. pixel_to_ground()   image position of the fire -> metres on the ground
                         (forward / right of the point directly below the drone)
  2. ground_to_pixel()   the inverse, so the predicted impact point can be drawn
                         on the pilot's video as a marker
  3. predict_impact()    where the ball lands, the miss distance, fire OK / wait
  4. ImpactGate          drop-in "confirm gate": needs the pilot's switch AND a
                         good, fresh impact prediction. Fails closed.
  5. velocity_heading_frame() / from_mavlink()  turn Pixhawk messages into the
                         inputs above

MODEL (and its honest limits)
  - Ground is flat and the altitude is height above it. get_relative_alt_m()
    is height above take-off, so use the TF-Luna rangefinder if the ground
    is not level.
  - The ball keeps the drone's horizontal velocity at the moment it leaves
    (so drift carries over), plus whatever the drone moves during
    `latency_s`, the time from the camera frame to the ball leaving the barrel.
  - Air drag and wind are ignored here (no wind sensor). monte_carlo_hit.py
    models them and shows how much they matter.
  - CAMERA FIELD OF VIEW AND MOUNTING ANGLE BELOW ARE PLACEHOLDERS. Set them
    to your real camera before trusting any number. Calibrate with a
    checkerboard if you want it precise.
  - Frames: x forward, y right, z down, heading-aligned (yaw removed).
    Pitch is positive nose-up, roll positive right-wing-down (ArduPilot).

Run the self-test:  python3 impact_predictor.py
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

try:
    from launcher_physics import time_of_flight_straight_down, exit_velocity_direct_drive
except ImportError:  # keeps this file usable on its own
    def exit_velocity_direct_drive(voltage: float = 14.8) -> float:
        return 4.117

    def time_of_flight_straight_down(altitude_m: float, v0: float) -> float:
        g = 9.81
        return (-v0 + math.sqrt(v0 * v0 + 2 * g * altitude_m)) / g


# ---------------------------------------------------------------------------
# Settings (change these to match your build)
# ---------------------------------------------------------------------------
HIT_RADIUS_M = 0.30          # what counts as a hit (same as the review slides)
LATENCY_S = 0.5              # PLACEHOLDER: camera frame -> ball leaves barrel. Measure it.
ALT_MIN_M, ALT_MAX_M = 2.0, 8.0   # same band as fire_detector.py's altitude gate


@dataclass
class CameraModel:
    hfov_deg: float = 62.2   # PLACEHOLDER (Pi Camera v2 is about 62.2 x 48.8): use yours
    vfov_deg: float = 48.8
    pitch_deg: float = 45.0  # PLACEHOLDER: angle of the optical axis below the horizon (90 = straight down)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _rot_level_from_body(roll_deg: float, pitch_deg: float):
    """Matrix taking a body-frame vector to the heading-aligned level frame."""
    p, r = math.radians(pitch_deg), math.radians(roll_deg)
    cp, sp, cr, sr = math.cos(p), math.sin(p), math.cos(r), math.sin(r)
    # level = Ry(pitch) @ Rx(roll) @ body
    return [
        [cp, sp * sr, sp * cr],
        [0.0, cr, -sr],
        [-sp, cp * sr, cp * cr],
    ]


def _mat_vec(m, v):
    return [sum(m[i][j] * v[j] for j in range(3)) for i in range(3)]


def _mat_t_vec(m, v):
    return [sum(m[j][i] * v[j] for j in range(3)) for i in range(3)]


def _camera_axes(cam: CameraModel):
    p = math.radians(cam.pitch_deg)
    fwd = (math.cos(p), 0.0, math.sin(p))
    right = (0.0, 1.0, 0.0)
    down = (-math.sin(p), 0.0, math.cos(p))
    return fwd, right, down


def pixel_to_ground(cx: float, cy: float, altitude_m: float, cam: CameraModel,
                    roll_deg: float = 0.0, pitch_deg: float = 0.0):
    """Normalised image position (0-1, origin top-left) -> (forward_m, right_m)
    of the ground point, measured from the spot directly below the drone.
    Returns None when the ray does not reach the ground (aimed above the horizon)."""
    u = (cx - 0.5) * 2.0 * math.tan(math.radians(cam.hfov_deg) / 2.0)
    v = (cy - 0.5) * 2.0 * math.tan(math.radians(cam.vfov_deg) / 2.0)
    f, r, d = _camera_axes(cam)
    body = [f[i] + u * r[i] + v * d[i] for i in range(3)]
    lvl = _mat_vec(_rot_level_from_body(roll_deg, pitch_deg), body)
    if lvl[2] <= 1e-9:
        return None
    s = altitude_m / lvl[2]
    return (s * lvl[0], s * lvl[1])


def ground_to_pixel(forward_m: float, right_m: float, altitude_m: float, cam: CameraModel,
                    roll_deg: float = 0.0, pitch_deg: float = 0.0):
    """Inverse of pixel_to_ground: where does this ground point appear in the
    image? Use it to draw the predicted impact marker. None if behind the camera."""
    body = _mat_t_vec(_rot_level_from_body(roll_deg, pitch_deg), [forward_m, right_m, altitude_m])
    f, r, d = _camera_axes(cam)
    depth = sum(body[i] * f[i] for i in range(3))
    if depth <= 1e-9:
        return None
    u = sum(body[i] * r[i] for i in range(3)) / depth
    v = sum(body[i] * d[i] for i in range(3)) / depth
    cx = 0.5 + u / (2.0 * math.tan(math.radians(cam.hfov_deg) / 2.0))
    cy = 0.5 + v / (2.0 * math.tan(math.radians(cam.vfov_deg) / 2.0))
    return (cx, cy)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

@dataclass
class ImpactPrediction:
    target_fwd: float       # fire, relative to the point below the drone now (m)
    target_right: float
    impact_fwd: float       # where the ball would land if released now (m)
    impact_right: float
    miss_m: float           # distance between the two
    fall_time_s: float
    fire_ok: bool
    reason: str
    correction_fwd: float   # how far to move the drone so the miss goes to zero
    correction_right: float
    timestamp: float = 0.0


def predict_impact(target_fwd: float, target_right: float,
                   v_fwd: float, v_right: float, altitude_m: float,
                   latency_s: float = LATENCY_S, radius_m: float = HIT_RADIUS_M,
                   v0: Optional[float] = None,
                   alt_band=(ALT_MIN_M, ALT_MAX_M)) -> ImpactPrediction:
    if v0 is None:
        v0 = exit_velocity_direct_drive()
    t_fall = time_of_flight_straight_down(altitude_m, v0)
    horizon = latency_s + t_fall
    impact_f, impact_r = v_fwd * horizon, v_right * horizon
    miss = math.hypot(impact_f - target_fwd, impact_r - target_right)
    lo, hi = alt_band
    if not (lo <= altitude_m <= hi):
        ok, reason = False, f"altitude {altitude_m:.1f} m outside {lo:.0f}-{hi:.0f} m band"
    elif miss <= radius_m:
        ok, reason = True, f"predicted miss {miss:.2f} m is inside {radius_m:.2f} m"
    else:
        ok, reason = False, f"predicted miss {miss:.2f} m is outside {radius_m:.2f} m"
    return ImpactPrediction(
        target_fwd=target_fwd, target_right=target_right,
        impact_fwd=impact_f, impact_right=impact_r, miss_m=miss, fall_time_s=t_fall,
        fire_ok=ok, reason=reason,
        correction_fwd=target_fwd - impact_f, correction_right=target_right - impact_r,
        timestamp=time.monotonic(),
    )


def predict_from_candidate(cx: float, cy: float, altitude_m: float, v_fwd: float, v_right: float,
                           cam: CameraModel, roll_deg: float = 0.0, pitch_deg: float = 0.0,
                           **kw) -> Optional[ImpactPrediction]:
    """Convenience: normalised candidate position -> prediction. None if the
    candidate cannot be placed on the ground."""
    g = pixel_to_ground(cx, cy, altitude_m, cam, roll_deg, pitch_deg)
    if g is None:
        return None
    return predict_impact(g[0], g[1], v_fwd, v_right, altitude_m, **kw)


def pilot_cue(p: Optional[ImpactPrediction], deadband_m: float = 0.1) -> str:
    """Short text for the pilot's video overlay."""
    if p is None:
        return "NO TARGET"
    if p.fire_ok:
        return "IN ZONE"
    parts = []
    if abs(p.correction_fwd) > deadband_m:
        parts.append(f"{'FWD' if p.correction_fwd > 0 else 'BACK'} {abs(p.correction_fwd):.1f} m")
    if abs(p.correction_right) > deadband_m:
        parts.append(f"{'RIGHT' if p.correction_right > 0 else 'LEFT'} {abs(p.correction_right):.1f} m")
    return "MOVE " + ", ".join(parts) if parts else "HOLD STILL"


# ---------------------------------------------------------------------------
# Confirm gate (fails closed). Same method as fire_detector.ConfirmGate.
# ---------------------------------------------------------------------------

class ImpactGate:
    """is_confirmed() is True only when (a) the base gate says yes (e.g. the
    pilot's RC switch) AND (b) the latest impact prediction says the shot is
    in the zone AND (c) that prediction is fresh. Anything missing -> False.

    USE (inside the main loop, once per frame):
        gate = ImpactGate(base_gate=RCSwitchConfirmGate(telemetry))
        ...
        gate.update(predict_from_candidate(...))   # or gate.update(None) if no target
        # then pass `gate` to MissionFSM wherever it takes a ConfirmGate
    """

    def __init__(self, base_gate=None, max_age_s: float = 0.5):
        self.base_gate = base_gate
        self.max_age_s = max_age_s
        self._pred: Optional[ImpactPrediction] = None

    def update(self, pred: Optional[ImpactPrediction]) -> None:
        self._pred = pred

    def is_confirmed(self) -> bool:
        p = self._pred
        if p is None or not p.fire_ok:
            return False
        if time.monotonic() - p.timestamp > self.max_age_s:
            return False
        if self.base_gate is not None and not self.base_gate.is_confirmed():
            return False
        return True


# ---------------------------------------------------------------------------
# MAVLink helpers (pure functions, so they can be tested without hardware)
# ---------------------------------------------------------------------------

def velocity_heading_frame(vx_cms: float, vy_cms: float, yaw_rad: float):
    """GLOBAL_POSITION_INT gives north/east velocity in cm/s. Returns
    (forward, right) in m/s for a drone facing `yaw_rad`."""
    vn, ve = vx_cms / 100.0, vy_cms / 100.0
    fwd = vn * math.cos(yaw_rad) + ve * math.sin(yaw_rad)
    right = -vn * math.sin(yaw_rad) + ve * math.cos(yaw_rad)
    return fwd, right


def from_mavlink(global_pos_msg, attitude_msg):
    """Returns (altitude_m, v_fwd, v_right, roll_deg, pitch_deg) from a
    GLOBAL_POSITION_INT and an ATTITUDE message (pymavlink objects, or anything
    with the same attributes). NOTE: written to the MAVLink message
    definitions but NOT tested against a real Pixhawk."""
    alt = global_pos_msg.relative_alt / 1000.0
    v_fwd, v_right = velocity_heading_frame(global_pos_msg.vx, global_pos_msg.vy, attitude_msg.yaw)
    return alt, v_fwd, v_right, math.degrees(attitude_msg.roll), math.degrees(attitude_msg.pitch)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _selftest():
    import random
    from types import SimpleNamespace
    from launcher_physics import required_hover_precision

    near = lambda a, b, tol=1e-9: abs(a - b) <= tol

    down = CameraModel(hfov_deg=60.0, vfov_deg=40.0, pitch_deg=90.0)

    # 1-2. Straight-down camera: centre pixel is directly below; edge pixel = h*tan(hfov/2).
    g = pixel_to_ground(0.5, 0.5, 5.0, down)
    assert near(g[0], 0.0) and near(g[1], 0.0), g
    g = pixel_to_ground(1.0, 0.5, 5.0, down)
    assert near(g[1], 5.0 * math.tan(math.radians(30.0))), g
    # 3. Camera tilted 45 degrees: centre pixel lands h/tan(45) ahead.
    g = pixel_to_ground(0.5, 0.5, 5.0, CameraModel(60, 40, 45.0))
    assert near(g[0], 5.0, 1e-9) and near(g[1], 0.0, 1e-9), g
    # 4. Looking above the horizon cannot hit the ground.
    assert pixel_to_ground(0.5, 0.0, 5.0, CameraModel(60, 48.8, 10.0)) is None
    # 5. Nose-up 10 degrees with a straight-down camera: ground point moves forward by h*tan(10).
    g = pixel_to_ground(0.5, 0.5, 5.0, down, pitch_deg=10.0)
    assert near(g[0], 5.0 * math.tan(math.radians(10.0)), 1e-9), g
    # 6. Round trip pixel -> ground -> pixel at random attitudes.
    rng = random.Random(1)
    cam = CameraModel(62.2, 48.8, 55.0)
    worst = 0.0
    n_checked = 0
    for _ in range(500):
        cx, cy = rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95)
        alt = rng.uniform(2, 8)
        roll, pitch = rng.uniform(-8, 8), rng.uniform(-8, 8)
        g = pixel_to_ground(cx, cy, alt, cam, roll, pitch)
        if g is None:
            continue
        back = ground_to_pixel(g[0], g[1], alt, cam, roll, pitch)
        worst = max(worst, abs(back[0] - cx), abs(back[1] - cy))
        n_checked += 1
    assert n_checked > 300 and worst < 1e-9, (n_checked, worst)

    # 7. Cross-check against the review slides: at the drift limit from
    #    launcher_physics the predicted miss is exactly the 0.30 m radius.
    v0 = exit_velocity_direct_drive()
    for alt in (2.0, 5.0, 8.0):
        vmax = required_hover_precision(alt, v0, 0.30)["max_tolerable_drift_velocity_mps"]
        at = predict_impact(0, 0, vmax, 0, alt, latency_s=0.0, radius_m=0.30 + 1e-9)
        over = predict_impact(0, 0, vmax * 1.02, 0, alt, latency_s=0.0)
        under = predict_impact(0, 0, vmax * 0.98, 0, alt, latency_s=0.0)
        assert near(at.miss_m, 0.30, 1e-6) and at.fire_ok, (alt, at)
        assert (not over.fire_ok) and under.fire_ok, (alt, over, under)
    # 8. Latency: at 0.1 m/s the ball lands 0.1 * (latency + fall time) ahead.
    p = predict_impact(0, 0, 0.1, 0, 5.0, latency_s=0.5)
    assert near(p.impact_fwd, 0.1 * (0.5 + p.fall_time_s), 1e-12)
    # 9. Altitude band blocks a firing that would otherwise be fine.
    assert not predict_impact(0, 0, 0, 0, 12.0).fire_ok
    # 10. Pilot cue and correction: target 1 m ahead, drone still -> move forward 1 m.
    p = predict_impact(1.0, 0.0, 0.0, 0.0, 5.0)
    assert (not p.fire_ok) and near(p.correction_fwd, 1.0) and pilot_cue(p).startswith("MOVE FWD 1.0 m")
    assert pilot_cue(predict_impact(0, 0, 0, 0, 5.0)) == "IN ZONE"
    assert pilot_cue(None) == "NO TARGET"
    # 11. MAVLink parsing: 1 m/s due north while facing east = 1 m/s to the LEFT.
    f, r = velocity_heading_frame(100.0, 0.0, math.pi / 2)
    assert near(f, 0.0, 1e-9) and near(r, -1.0, 1e-9), (f, r)
    gp = SimpleNamespace(relative_alt=5000, vx=0, vy=0)
    at_ = SimpleNamespace(yaw=0.0, roll=0.0, pitch=math.radians(3))
    alt, vf, vr, roll, pitch = from_mavlink(gp, at_)
    assert near(alt, 5.0) and near(pitch, 3.0, 1e-9)

    # 12. ImpactGate fails closed in every missing-information case.
    class Switch:
        def __init__(self, on): self.on = on
        def is_confirmed(self): return self.on
    good = predict_impact(0, 0, 0, 0, 5.0)
    bad = predict_impact(1.0, 0, 0, 0, 5.0)
    gate = ImpactGate(Switch(True), max_age_s=0.2)
    assert gate.is_confirmed() is False                    # no prediction yet
    gate.update(good);       assert gate.is_confirmed() is True
    gate.update(bad);        assert gate.is_confirmed() is False   # miss too big
    gate.update(None);       assert gate.is_confirmed() is False   # target lost
    gate.update(good)
    assert ImpactGate(Switch(False)).is_confirmed() is False
    g2 = ImpactGate(Switch(False)); g2.update(good); assert g2.is_confirmed() is False  # pilot switch off
    gate.update(good); time.sleep(0.25); assert gate.is_confirmed() is False            # stale prediction

    print("SELFTEST PASSED: geometry (incl. round trip over %d random cases), impact maths "
          "cross-checked against launcher_physics drift limits, and gate fails closed." % n_checked)


if __name__ == "__main__":
    _selftest()
