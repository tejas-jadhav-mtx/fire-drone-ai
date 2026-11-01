"""
Image-folder check of the fire detector (your own photos)
----------------------------------------------------------------------------
Use this when you have PICTURES instead of video. Put them in two folders:

    my_images/
        fire/      <- pictures that contain fire (flames visible)
        nofire/    <- pictures with NO fire: sunsets, orange objects, lamps,
                      LEDs, tail lights, high-visibility jackets, bonfire-coloured
                      walls, anything the detector might mistake for fire

Run:
    python eval_images.py --folder my_images --weights best.pt

It reports, for several confidence thresholds:
  - fire pictures where the detector said fire  (recall)
  - no-fire pictures where the detector said fire  (false alarms)
and lists the no-fire pictures that fooled it hardest, so you can see WHY.

Pictures cannot test the 7-of-10 confirmation rule (that needs video), so this
answers only "does it say fire on the right pictures?". Many different no-fire
pictures (50-100) are the cheap way to find false alarms.

Self-test (no pictures needed):  python eval_images.py --selftest
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

THRESHOLDS = (0.25, 0.30, 0.40, 0.50, 0.60)
EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")


def list_images(folder: str):
    out = []
    for e in EXTS:
        out += glob.glob(os.path.join(folder, e)) + glob.glob(os.path.join(folder, e.upper()))
    return sorted(set(out))


def summarise(fire_scores: dict, nofire_scores: dict, thresholds=THRESHOLDS):
    """fire_scores / nofire_scores: {filename: highest fire confidence in that picture (0 if none)}.
    Returns [(threshold, recall, false_alarm_rate, n_fire_hit, n_nofire_hit)]."""
    rows = []
    for t in thresholds:
        fh = sum(s >= t for s in fire_scores.values())
        nh = sum(s >= t for s in nofire_scores.values())
        rec = fh / len(fire_scores) if fire_scores else None
        fa = nh / len(nofire_scores) if nofire_scores else None
        rows.append((t, rec, fa, fh, nh))
    return rows


def pct(x):
    return "-" if x is None else f"{100 * x:.0f}%"


def report(fire_scores, nofire_scores):
    print()
    print(f"{len(fire_scores)} fire pictures, {len(nofire_scores)} no-fire pictures")
    print(f"{'threshold':>10}{'fire found':>14}{'false alarms':>16}")
    for t, rec, fa, fh, nh in summarise(fire_scores, nofire_scores):
        print(f"{t:>10.2f}{f'{fh}/{len(fire_scores)} ({pct(rec)})':>14}{f'{nh}/{len(nofire_scores)} ({pct(fa)})':>16}")
    bad = sorted(nofire_scores.items(), key=lambda kv: -kv[1])[:8]
    bad = [(n, s) for n, s in bad if s >= 0.25]
    if bad:
        print("\nNo-fire pictures that fooled the detector most (highest fire confidence):")
        for n, s in bad:
            print(f"   {s:.2f}  {n}")
    miss = sorted(fire_scores.items(), key=lambda kv: kv[1])[:8]
    miss = [(n, s) for n, s in miss if s < 0.4]
    if miss:
        print("\nFire pictures it was least sure about:")
        for n, s in miss:
            print(f"   {s:.2f}  {n}")
    if len(fire_scores) < 20 or len(nofire_scores) < 30:
        print("\nNOTE: with this few pictures the percentages are rough; a single picture moves them a lot.")


def score_folder(folder, model, imgsz):
    import cv2
    import yolo_fire_detector as yf
    scores = {}
    for p in list_images(folder):
        img = cv2.imread(p)
        if img is None:
            continue
        r = model.model.predict(img, imgsz=imgsz, conf=0.05, verbose=False)[0]
        best = max([float(b.conf) for b in r.boxes if int(b.cls) in yf.FIRE_CLASS_IDS], default=0.0)
        scores[os.path.basename(p)] = best
    return scores


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder")
    ap.add_argument("--weights")
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    if not (a.folder and a.weights):
        ap.error("--folder and --weights are required")
    import yolo_fire_detector as yf
    model = yf.YoloFireModel(a.weights, imgsz=a.imgsz)
    fire = score_folder(os.path.join(a.folder, "fire"), model, a.imgsz)
    nofire = score_folder(os.path.join(a.folder, "nofire"), model, a.imgsz)
    if not fire and not nofire:
        raise SystemExit(f"no pictures found in {a.folder}/fire or {a.folder}/nofire")
    report(fire, nofire)


def _selftest():
    fire = {"a.jpg": 0.9, "b.jpg": 0.45, "c.jpg": 0.2, "d.jpg": 0.7}
    nofire = {"x.jpg": 0.0, "y.jpg": 0.33, "z.jpg": 0.0, "w.jpg": 0.55}
    rows = {t: (rec, fa, fh, nh) for t, rec, fa, fh, nh in summarise(fire, nofire)}
    assert rows[0.25][2:] == (3, 2)            # 0.9, 0.45, 0.7 are >= 0.25; y and w
    assert rows[0.30][2:] == (3, 2)
    assert rows[0.40][2:] == (3, 1)            # only w (0.55) fools it at 0.40
    assert rows[0.50][2:] == (2, 1)
    assert rows[0.60][2:] == (2, 0)            # 0.9 and 0.7 clear 0.60; the 0.55 no-fire picture does not
    assert abs(rows[0.40][0] - 0.75) < 1e-12 and abs(rows[0.40][1] - 0.25) < 1e-12
    assert summarise({}, nofire)[0][1] is None  # no fire pictures -> recall undefined, not a crash
    import tempfile, numpy as np, cv2
    d = tempfile.mkdtemp()
    cv2.imwrite(os.path.join(d, "1.jpg"), np.zeros((8, 8, 3), np.uint8))
    cv2.imwrite(os.path.join(d, "2.PNG"), np.zeros((8, 8, 3), np.uint8))
    open(os.path.join(d, "notes.txt"), "w").write("x")
    assert len(list_images(d)) == 2
    print("SELFTEST PASSED: threshold sweep, recall/false-alarm arithmetic, empty-class handling, image listing")


if __name__ == "__main__":
    main()
