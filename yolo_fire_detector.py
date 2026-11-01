"""
Drop-in YOLOv8-based fire/smoke detector -- same Candidate contract as
detect_fire_color() in fire_detector.py, so it can replace (or run
alongside, for comparison) the classical HSV detector without touching
MissionFSM, the ESP8266 firmware, or anything else in the pipeline.

Needs a trained weights file (best.pt) from train_fire_yolo.py -- this
module does not train anything itself, only runs inference.

USAGE (inside fire_detector.py's MissionFSM.step(), swap one line):
    # before:
    optical_c = detect_fire_color(frame) if frame is not None else []
    # after:
    optical_c = detect_fire_yolo(frame, yolo_model) if frame is not None else []
(and construct `yolo_model = YoloFireModel("best.pt")` once, alongside the
other sensor objects in main())
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# Imported lazily inside the class below, matching the pattern the rest of
# this project already uses (MLX90640Thermal, PymavlinkTelemetry, etc.) so
# this file can still be imported/inspected on a machine without
# ultralytics installed.


class Candidate:
    """Duplicated here on purpose (rather than importing it from
    fire_detector.py) so this file has zero import-time dependency on
    that module -- you can syntax-check and reason about this file in
    isolation. If you import both in the same program, make sure you
    only construct ONE Candidate class consistently (e.g., import the
    real one from fire_detector and delete this duplicate) -- two
    separate classes with the same name and fields are interchangeable
    in practice (duck typing) but a type checker would complain."""
    def __init__(self, cx: float, cy: float, area: float, strength: float):
        self.cx = cx
        self.cy = cy
        self.area = area
        self.strength = strength


# Must match whatever class order your FASDD export used -- check
# data.yaml's `names:` list after download_dataset() in
# train_fire_yolo.py and update this if it differs.
CLASS_NAMES = {0: "fire", 1: "smoke"}
FIRE_CLASS_IDS = {0}  # which of the above actually counts as "fire" for
                      # this project -- smoke alone, with no confirmed
                      # flame, is deliberately NOT enough to arm the
                      # launcher; see CONFIDENCE_THRESHOLD below
# 0.4, not 0.5: image-level test on 3,000 validation images (1008 fire, 743
# smoke-only, 1249 background) gave recall 0.84 / false alarms 1.1% at 0.4,
# versus recall 0.71 at 0.5. The 7-of-10 frame rule in fire_detector.py
# already suppresses isolated false frames, so a lower per-frame threshold is
# affordable. Dataset images, not drone footage: re-check on your own clips.
CONFIDENCE_THRESHOLD = 0.4
DEFAULT_IMGSZ = 416   # the size the model was trained at; 320 is faster but unverified for recall


class YoloFireModel:
    def __init__(self, weights_path: str, imgsz: int = DEFAULT_IMGSZ):
        from ultralytics import YOLO  # lazy import, see module docstring
        self.model = YOLO(weights_path)
        self.imgsz = imgsz

    def infer(self, frame_bgr: np.ndarray):
        # conf is passed so the model drops weak boxes early; the class filter
        # and the same threshold are applied again in detect_fire_yolo().
        return self.model.predict(frame_bgr, imgsz=self.imgsz, conf=CONFIDENCE_THRESHOLD,
                                  verbose=False)[0]


def detect_fire_yolo(frame_bgr: np.ndarray, model: YoloFireModel) -> list[Candidate]:
    """Same signature/contract as detect_fire_color(): takes a BGR frame,
    returns a list of Candidate in NORMALIZED (0-1) coordinates."""
    h, w = frame_bgr.shape[:2]
    result = model.infer(frame_bgr)

    out = []
    for box in result.boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        if cls_id not in FIRE_CLASS_IDS or conf < CONFIDENCE_THRESHOLD:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        area = (x2 - x1) * (y2 - y1)
        out.append(Candidate(cx=cx, cy=cy, area=area, strength=conf))
    return out
