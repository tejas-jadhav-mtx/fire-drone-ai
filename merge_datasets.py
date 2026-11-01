"""
Merge several YOLO-format fire/smoke datasets into one training set
----------------------------------------------------------------------------
Every source dataset names its classes differently (FASDD lost its names and
only has '0' and '1'; D-Fire has an extra 'object' class). This module maps
every source onto ONE class list:

        0 = fire      1 = smoke

and drops everything else. Pictures whose boxes were all dropped (or that had
no boxes to begin with) are KEPT as empty-label pictures: those are the
"hard negatives" that teach the detector what is NOT fire.

Every output file name starts with the source name (e.g. dfire__img123.jpg) so
results can later be broken down per source.

Used by train_fire_detector_v2.ipynb. Self-test:  python merge_datasets.py
"""

from __future__ import annotations

import glob
import os
import random
import shutil
from typing import Dict, List, Optional

import yaml

CLASS_NAMES = ["fire", "smoke"]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SPLIT_ALIASES = {"train": ("train",), "valid": ("valid", "val", "validation")}


# ---------------------------------------------------------------------------
# Class mapping
# ---------------------------------------------------------------------------

def read_names(root: str) -> List[str]:
    """Class names from root/data.yaml, as a list ordered by class id."""
    cfg = yaml.safe_load(open(os.path.join(root, "data.yaml")))
    names = cfg.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]
    return [str(n) for n in names]


def auto_map(names: List[str]) -> Optional[Dict[int, Optional[int]]]:
    """Guess source-class -> {0 fire, 1 smoke, None drop} from the NAMES.
    Returns None when the names say nothing (e.g. '0','1'): the caller must
    then give the mapping by hand after looking at pictures."""
    if all(n.strip().isdigit() for n in names):
        return None
    out: Dict[int, Optional[int]] = {}
    for i, n in enumerate(names):
        low = n.lower()
        if "smoke" in low:
            out[i] = 1
        elif "fire" in low or "flame" in low:
            out[i] = 0
        else:
            out[i] = None
    if 0 not in out.values():
        return None          # no class we can call fire: refuse to guess
    return out


# ---------------------------------------------------------------------------
# Label conversion
# ---------------------------------------------------------------------------

def convert_label_text(text: str, class_map: Dict[int, Optional[int]]):
    """Returns (new_text, kept_per_class{0,1}, n_dropped). Handles box lines
    ('c x y w h') and polygon lines ('c x1 y1 x2 y2 ...', turned into a box)."""
    kept = {0: 0, 1: 0}
    dropped = 0
    lines = []
    for line in text.splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        try:
            c = int(float(p[0]))
            vals = [float(v) for v in p[1:]]
        except ValueError:
            continue
        dst = class_map.get(c)
        if dst is None:
            dropped += 1
            continue
        if len(vals) == 4:
            x, y, w, h = vals
        else:                                   # polygon -> bounding box
            xs, ys = vals[0::2], vals[1::2]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            x, y, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
        if w <= 0 or h <= 0:
            dropped += 1
            continue
        x, y = min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)
        w, h = min(w, 1.0), min(h, 1.0)
        lines.append(f"{dst} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        kept[dst] += 1
    return ("\n".join(lines) + ("\n" if lines else ""), kept, dropped)


# ---------------------------------------------------------------------------
# Finding pictures
# ---------------------------------------------------------------------------

def split_dirs(root: str, split: str):
    """(images_dir, labels_dir) for 'train' or 'valid', or None if absent."""
    for alias in SPLIT_ALIASES[split]:
        im = os.path.join(root, alias, "images")
        if os.path.isdir(im):
            return im, os.path.join(root, alias, "labels")
    return None


def list_pairs(images_dir: str, labels_dir: str):
    pairs = []
    for p in sorted(glob.glob(os.path.join(images_dir, "*"))):
        if os.path.splitext(p)[1].lower() not in IMG_EXTS:
            continue
        stem = os.path.splitext(os.path.basename(p))[0]
        lp = os.path.join(labels_dir, stem + ".txt")
        pairs.append((p, lp if os.path.exists(lp) else None))
    return pairs


def _kept_boxes(lp, class_map):
    if lp is None:
        return 0
    k = convert_label_text(open(lp).read(), class_map)[1]
    return k[0] + k[1]


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge(sources: List[dict], out_dir: str, seed: int = 1) -> dict:
    """sources: dicts with
         name, root, class_map (None -> auto_map), train_cap, val_cap (None = all),
         max_empty_frac (largest share of empty pictures in what is taken, e.g. 0.4)
       Writes out_dir/{train,valid}/{images,labels} and out_dir/data.yaml.
       Returns statistics per source and split."""
    rng = random.Random(seed)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    stats: dict = {}
    for s in sources:
        name, root = s["name"], s["root"]
        cmap = s.get("class_map")
        if cmap is None:
            cmap = auto_map(read_names(root))
            if cmap is None:
                raise ValueError(f"{name}: class names are not informative; give class_map by hand "
                                 f"(look at pictures with preview_grid first)")
        cmap = {int(k): v for k, v in cmap.items()}
        for split in ("train", "valid"):
            d = split_dirs(root, split)
            st = stats.setdefault(name, {}).setdefault(
                split, {"images": 0, "empty": 0, "fire_boxes": 0, "smoke_boxes": 0, "dropped_boxes": 0})
            if d is None:
                continue
            pairs = list_pairs(*d)
            pos = [p for p in pairs if _kept_boxes(p[1], cmap) > 0]
            emp = [p for p in pairs if _kept_boxes(p[1], cmap) == 0]
            rng.shuffle(pos)
            rng.shuffle(emp)
            cap = s.get(f"{'train' if split == 'train' else 'val'}_cap")
            if cap is None:
                chosen = pos + emp
                mef = s.get("max_empty_frac")
                if mef is not None and pos:
                    # at most mef of the final set is empty: e <= mef/(1-mef) * len(pos)
                    emp_cap = int(round(mef / (1 - mef) * len(pos))) if mef < 1 else len(emp)
                    chosen = pos + emp[:emp_cap]
            else:
                mef = s.get("max_empty_frac")
                n_emp = min(len(emp), int(cap * mef) if mef is not None else len(emp))
                n_pos = min(len(pos), cap - n_emp)
                n_emp = min(len(emp), cap - n_pos)     # give unused room back to empties
                chosen = pos[:n_pos] + emp[:n_emp]
            oi = os.path.join(out_dir, split, "images")
            ol = os.path.join(out_dir, split, "labels")
            os.makedirs(oi, exist_ok=True)
            os.makedirs(ol, exist_ok=True)
            for ip, lp in chosen:
                base = f"{name}__{os.path.basename(ip)}"
                stem = os.path.splitext(base)[0]
                shutil.copy(ip, os.path.join(oi, base))
                if lp is not None:
                    txt, kept, dropped = convert_label_text(open(lp).read(), cmap)
                else:
                    txt, kept, dropped = "", {0: 0, 1: 0}, 0
                open(os.path.join(ol, stem + ".txt"), "w").write(txt)
                st["images"] += 1
                st["empty"] += int(kept[0] + kept[1] == 0)
                st["fire_boxes"] += kept[0]
                st["smoke_boxes"] += kept[1]
                st["dropped_boxes"] += dropped
    cfg = {"path": os.path.abspath(out_dir), "train": "train/images", "val": "valid/images",
           "nc": 2, "names": CLASS_NAMES}
    yaml.safe_dump(cfg, open(os.path.join(out_dir, "data.yaml"), "w"))
    return stats


def print_stats(stats: dict):
    print(f"{'source':<10}{'split':<7}{'images':>8}{'empty':>8}{'fire boxes':>12}{'smoke boxes':>13}{'dropped':>9}")
    tot = {"train": 0, "valid": 0}
    for name, sp in stats.items():
        for split, st in sp.items():
            print(f"{name:<10}{split:<7}{st['images']:>8}{st['empty']:>8}{st['fire_boxes']:>12}"
                  f"{st['smoke_boxes']:>13}{st['dropped_boxes']:>9}")
            tot[split] += st["images"]
    print(f"TOTAL train {tot['train']}, valid {tot['valid']}")


# ---------------------------------------------------------------------------
# Look at labels (do this once per source with unknown class order)
# ---------------------------------------------------------------------------

def preview_grid(root: str, class_map: Optional[dict], n: int = 8, seed: int = 1, split: str = "train"):
    """Draws n random pictures with their boxes. Colour = ORIGINAL class id of the
    source (0 orange, 1 blue, 2 green, 3 magenta, others grey) so you can see which
    id sits on flames. Returns a matplotlib figure."""
    import cv2
    import matplotlib.pyplot as plt
    d = split_dirs(root, split)
    pairs = [p for p in list_pairs(*d) if p[1] and os.path.getsize(p[1]) > 0]
    random.Random(seed).shuffle(pairs)
    cols = {0: (255, 140, 0), 1: (0, 120, 255), 2: (0, 200, 0), 3: (220, 0, 220)}
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, (ip, lp) in zip(axes.ravel(), pairs[:n]):
        img = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        seen = set()
        for line in open(lp):
            p = line.split()
            if len(p) < 5:
                continue
            c = int(float(p[0]))
            vals = [float(v) for v in p[1:]]
            if len(vals) == 4:
                x, y, bw, bh = vals
            else:
                xs, ys = vals[0::2], vals[1::2]
                x, y, bw, bh = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)
            x1, y1, x2, y2 = int((x - bw / 2) * w), int((y - bh / 2) * h), int((x + bw / 2) * w), int((y + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), cols.get(c, (128, 128, 128)), max(2, w // 300))
            seen.add(c)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title("source classes in picture: " + ",".join(map(str, sorted(seen))), fontsize=9)
    fig.suptitle("orange = source class 0, blue = 1, green = 2, magenta = 3", fontsize=12)
    return fig


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _selftest():
    import tempfile
    import numpy as np
    import cv2

    # --- auto_map
    assert auto_map(["fire", "smoke"]) == {0: 0, 1: 1}
    assert auto_map(["smoke", "fire"]) == {0: 1, 1: 0}
    assert auto_map(["fire", "smoke", "object"]) == {0: 0, 1: 1, 2: None}
    assert auto_map(["0", "1"]) is None                     # unnamed: must be given by hand
    assert auto_map(["smoke", "person"]) is None            # nothing fire-like: refuse to guess
    assert auto_map(["Fire-Flame"]) == {0: 0}

    # --- label conversion
    txt = "0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n2 0.1 0.1 0.1 0.1\n0 0.5 0.5 0 0.2\n"
    out, kept, dropped = convert_label_text(txt, {0: 0, 1: 1, 2: None})
    assert kept == {0: 1, 1: 1} and dropped == 2 and out.count("\n") == 2
    out, kept, _ = convert_label_text(txt, {0: 1, 1: 0, 2: None})          # swapped order
    assert out.splitlines()[0].startswith("1 ") and out.splitlines()[1].startswith("0 ")
    poly = "0 0.2 0.2 0.6 0.2 0.6 0.8 0.2 0.8\n"                            # polygon -> box
    out, kept, _ = convert_label_text(poly, {0: 0})
    x, y, w, h = [float(v) for v in out.split()[1:]]
    assert abs(x - 0.4) < 1e-6 and abs(y - 0.5) < 1e-6 and abs(w - 0.4) < 1e-6 and abs(h - 0.6) < 1e-6
    out, kept, _ = convert_label_text("0 1.2 -0.1 0.3 0.3\n", {0: 0})      # out-of-range centre is clamped
    assert all(0 <= float(v) <= 1 for v in out.split()[1:])

    # --- build two fake datasets in Roboflow layout
    tmp = tempfile.mkdtemp()

    def make(root, names, n_train, n_valid, label_fn):
        for split in ("train", "valid"):
            os.makedirs(f"{root}/{split}/images")
            os.makedirs(f"{root}/{split}/labels")
            for i in range(n_train if split == "train" else n_valid):
                cv2.imwrite(f"{root}/{split}/images/im{i}.jpg", np.full((32, 32, 3), i % 255, np.uint8))
                t = label_fn(i)
                if t is not None:
                    open(f"{root}/{split}/labels/im{i}.txt", "w").write(t)
        yaml.safe_dump({"names": names, "nc": len(names)}, open(f"{root}/data.yaml", "w"))

    # A: names lost ('0','1'); 60 train: 40 with fire, 10 with smoke, 10 empty; 20 valid like it
    A = f"{tmp}/A"
    make(A, ["0", "1"], 60, 20,
         lambda i: "0 0.5 0.5 0.3 0.3\n" if i % 3 else ("1 0.4 0.4 0.3 0.3\n" if i % 6 == 0 else ""))
    # B: D-Fire style: smoke=0, fire=1, object=2; 100 train: 20 fire, 30 only 'object' (-> negatives), 50 no label file
    B = f"{tmp}/B"
    make(B, ["smoke", "fire", "object"], 100, 10,
         lambda i: "1 0.5 0.5 0.2 0.2\n" if i < 20 else ("2 0.5 0.5 0.2 0.2\n" if i < 50 else None))

    sources = [
        dict(name="a", root=A, class_map={0: 0, 1: 1}, train_cap=None, val_cap=None),
        dict(name="b", root=B, class_map=None, train_cap=60, val_cap=None, max_empty_frac=0.5),
    ]
    out = os.path.join(tmp, "merged")
    stats = merge(sources, out, seed=3)
    a, b = stats["a"]["train"], stats["b"]["train"]
    assert a["images"] == 60 and a["empty"] == 10                                # nothing dropped without caps
    assert b["images"] == 60 and b["fire_boxes"] == 20 and b["smoke_boxes"] == 0
    assert b["empty"] == 40, b                                                  # 20 positives + 40 negatives (cap 60)
    assert b["dropped_boxes"] > 0                                               # 'object' boxes were dropped
    # B was auto-mapped from names: its 'fire' (id 1) must now be class 0
    lab = [f for f in glob.glob(f"{out}/train/labels/b__*.txt") if open(f).read().strip()]
    assert lab and all(open(f).read().split()[0] == "0" for f in lab)
    # file names carry the source prefix and every image has a label file
    ims = sorted(os.listdir(f"{out}/train/images"))
    lbs = sorted(os.listdir(f"{out}/train/labels"))
    assert all(n.startswith(("a__", "b__")) for n in ims)
    assert [os.path.splitext(i)[0] for i in ims] == [os.path.splitext(l)[0] for l in lbs]
    cfg = yaml.safe_load(open(f"{out}/data.yaml"))
    assert cfg["names"] == ["fire", "smoke"] and cfg["train"] == "train/images" and cfg["val"] == "valid/images"
    # a cap smaller than the positives is respected; empties never exceed their share
    stats2 = merge([dict(name="a", root=A, class_map={0: 0, 1: 1}, train_cap=30, val_cap=5,
                         max_empty_frac=0.2)], out, seed=3)
    s2 = stats2["a"]["train"]
    assert s2["images"] == 30 and s2["empty"] <= 6, s2
    assert stats2["a"]["valid"]["images"] == 5
    # unnamed classes without a hand mapping must be refused, not guessed
    try:
        merge([dict(name="a", root=A, class_map=None, train_cap=5, val_cap=5)], out)
        raise AssertionError("should have refused")
    except ValueError:
        pass
    # same seed -> same selection (reproducible)
    merge(sources, out, seed=5); first = sorted(os.listdir(f"{out}/train/images"))
    merge(sources, out, seed=5); assert first == sorted(os.listdir(f"{out}/train/images"))
    print_stats(stats)
    print("SELFTEST PASSED: class auto-mapping, polygon->box, clamping, negatives kept, caps, reproducibility, refusal to guess")


if __name__ == "__main__":
    _selftest()
