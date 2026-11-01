# Fire-Drone AI: fire detection and launch decision on a Raspberry Pi 4

Onboard software for my B.Tech capstone: a **fire-station drone first responder** that spots a fire from the air and fires a suppression ball at it from a turret launcher. The drone is flown by a human pilot over RC. This code owns everything from *"is there a fire in this frame?"* to *"is it safe and accurate to release the ball right now?"*.

> **Status:** prototype stage. Every module has a self-test that runs on a laptop with no hardware attached. The detection model has been trained; field tests on the drone are still to come.

## How it works

```
camera frame ──► fire detector (YOLOv8n or HSV colour baseline)
                    │
                    ▼
              Kalman target tracker  ──► fire confirmed in 7 of the last 10 frames
                    │
                    ▼
              impact predictor  ──► where would the ball land if released now?
                    │                (altitude, drone velocity, camera geometry)
                    ▼
              firing gate  ──► pilot's RC confirm switch AND predicted miss < 0.3 m
                    │
                    ▼
              "FIRE" over serial ──► NodeMCU (ESP8266) runs the launcher motors
```

**Safety first:** the default mode is semi-autonomous. The software detects and confirms a fire on its own, but it will not fire without the pilot's confirmation, and the firing gate fails closed when any input is missing or stale.

## Files

| File | What it does |
|---|---|
| `fire_detector.py` | Main mission state machine: detection, temporal confirmation, firing gate, serial command to the launcher |
| `yolo_fire_detector.py` | YOLOv8 fire/smoke detector, a drop-in replacement for the HSV colour detector |
| `target_tracker.py` | Constant-velocity Kalman tracker with Mahalanobis gating, M-of-N confirmation and coasting |
| `impact_predictor.py` | Pixel → ground projection, ball impact prediction, pilot steering cue, firing gate |
| `launcher_physics.py` | Flywheel RPM → ball exit velocity → trajectory; drift limits for a 0.3 m miss |
| `merge_datasets.py` | Merges several YOLO fire/smoke datasets onto one class list, keeping hard negatives |
| `eval_images.py` | Tests the detector on your own `fire/` and `nofire/` photos (sunsets, lamps, hi-vis jackets) |
| `notebooks/train_fire_detector_v2.ipynb` | Colab notebook: merge datasets, fine-tune YOLOv8n, compare v1 vs v2, export to NCNN for the Pi |
| `models/fire_smoke_yolov8n_v1.pt` | v1 fire/smoke weights (YOLOv8n) |

## Run the self-tests (no hardware needed)

```bash
pip install -r requirements.txt
python fire_detector.py --selftest
python target_tracker.py --selftest
python impact_predictor.py
python launcher_physics.py
python merge_datasets.py
python eval_images.py --selftest
```

Test images of fire-like objects:

```bash
python eval_images.py --folder my_images --weights models/fire_smoke_yolov8n_v1.pt
```

## Hardware (prototype)

Raspberry Pi 4B (4 GB) with Pi Camera Module 3 · NodeMCU (ESP8266) for the launcher · twin counter-rotating flywheels with an 80 A dual brushed ESC · NEMA17 + A4988 for side-to-side aiming · MG90S servo ball lock · BotWing F405 flight controller · 6S 5200 mAh LiPo · 730 mm frame, 3D-printed in ABS.

## Notes and limits

- The code also supports an optional MLX90640 thermal camera and MAVLink telemetry. The current prototype is camera-only, so those paths are tested only with the built-in fakes.
- `launcher_physics.py` uses motor figures from an earlier parts list. Re-check them against the final launcher motors before relying on its numbers.
- The NodeMCU launcher firmware is not in this repository yet.
- The impact model ignores air drag and wind. Camera field of view and mounting angle are placeholders until the camera is calibrated.
- Never test with real fire. Use a photo or video of fire on a screen.

**Author:** Tejas Jadhav, B.Tech Mechatronics, Symbiosis Skills and Professional University, Pune
