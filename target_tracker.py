"""
target_tracker.py -- Kalman tracking of fire candidates (idea 4)
================================================================
Turns a per-frame list of detections (cx, cy, area in 0-1 coordinates, like
fire_detector.Candidate) into persistent TRACKS.

  * Constant-velocity Kalman filter on (cx, cy) -> smooth position + velocity
  * Mahalanobis-gated nearest-neighbour association (no scipy needed)
  * M-of-N confirmation (same 7-of-10 rule as fire_detector CONFIRM_COUNT /
    CONFIRM_WINDOW) per track, not per frame
  * Coasting: a track survives short dropouts and keeps predicting
  * Flicker score: relative variation of blob area over the recent window.
    This is an UNVALIDATED hypothesis (real flames flicker, orange cloth
    does not). It is reported only; nothing here uses it for decisions.

Verified only on synthetic data (see --selftest). Noise values are
assumptions until tuned on real footage.
"""
from __future__ import annotations
import math
from collections import deque
from dataclasses import dataclass, field
import numpy as np

CONFIRM_WINDOW = 10
CONFIRM_COUNT = 7
MAX_MISSES = 8          # frames a track may coast before deletion
GATE_CHI2 = 9.21        # 99% for 2 dof
SIGMA_ACCEL = 0.3       # process noise, normalised units / s^2  (assumption)
SIGMA_MEAS = 0.02       # measurement noise, normalised units    (assumption)


@dataclass
class Track:
    tid: int
    x: np.ndarray                 # [cx, cy, vx, vy]
    P: np.ndarray
    hits: deque = field(default_factory=lambda: deque(maxlen=CONFIRM_WINDOW))
    areas: deque = field(default_factory=lambda: deque(maxlen=CONFIRM_WINDOW))
    misses: int = 0
    confirmed: bool = False

    @property
    def pos(self): return float(self.x[0]), float(self.x[1])
    @property
    def vel(self): return float(self.x[2]), float(self.x[3])

    @property
    def flicker(self) -> float:
        a = np.array(self.areas, dtype=float)
        if len(a) < 4 or a.mean() <= 0:
            return 0.0
        return float(a.std() / a.mean())


class TargetTracker:
    def __init__(self, sigma_accel=SIGMA_ACCEL, sigma_meas=SIGMA_MEAS,
                 confirm_window=CONFIRM_WINDOW, confirm_count=CONFIRM_COUNT,
                 max_misses=MAX_MISSES, gate_chi2=GATE_CHI2):
        self.qa, self.sm = sigma_accel, sigma_meas
        self.cw, self.cc = confirm_window, confirm_count
        self.max_misses, self.gate = max_misses, gate_chi2
        self.tracks: list[Track] = []
        self._next = 1

    # --- model ---
    def _F(self, dt):
        F = np.eye(4); F[0, 2] = F[1, 3] = dt; return F

    def _Q(self, dt):
        q = self.qa ** 2
        b = np.array([[dt**4/4, dt**3/2], [dt**3/2, dt**2]]) * q
        Q = np.zeros((4, 4))
        for i in range(2):
            Q[i, i], Q[i, i+2], Q[i+2, i], Q[i+2, i+2] = b[0,0], b[0,1], b[1,0], b[1,1]
        return Q

    H = np.array([[1., 0, 0, 0], [0, 1., 0, 0]])

    def _predict(self, t: Track, dt):
        F = self._F(dt)
        t.x = F @ t.x
        t.P = F @ t.P @ F.T + self._Q(dt)

    def _update(self, t: Track, z):
        R = np.eye(2) * self.sm ** 2
        S = self.H @ t.P @ self.H.T + R
        K = t.P @ self.H.T @ np.linalg.inv(S)
        t.x = t.x + K @ (z - self.H @ t.x)
        t.P = (np.eye(4) - K @ self.H) @ t.P

    def _d2(self, t: Track, z):
        S = self.H @ t.P @ self.H.T + np.eye(2) * self.sm ** 2
        r = z - self.H @ t.x
        return float(r @ np.linalg.inv(S) @ r)

    # --- main step ---
    def step(self, detections, dt: float):
        """detections: iterable of objects with .cx .cy .area. Returns tracks."""
        dets = list(detections)
        for t in self.tracks:
            self._predict(t, dt)

        pairs = []
        for ti, t in enumerate(self.tracks):
            for di, d in enumerate(dets):
                d2 = self._d2(t, np.array([d.cx, d.cy]))
                if d2 <= self.gate:
                    pairs.append((d2, ti, di))
        pairs.sort()
        used_t, used_d = set(), set()
        for d2, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti); used_d.add(di)
            t, d = self.tracks[ti], dets[di]
            self._update(t, np.array([d.cx, d.cy]))
            t.misses = 0
            t.hits.append(1)
            t.areas.append(d.area)

        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.misses += 1
                t.hits.append(0)

        for di, d in enumerate(dets):
            if di in used_d:
                continue
            P = np.diag([self.sm**2, self.sm**2, 1.0, 1.0])
            t = Track(self._next, np.array([d.cx, d.cy, 0., 0.]), P,
                      deque(maxlen=self.cw), deque(maxlen=self.cw))
            self._next += 1
            t.hits.append(1); t.areas.append(d.area)
            self.tracks.append(t)

        for t in self.tracks:
            if sum(t.hits) >= self.cc:
                t.confirmed = True
            elif t.confirmed and sum(t.hits) < self.cc and t.misses > 0:
                pass  # stays confirmed until deleted (hysteresis)
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        return self.tracks

    def confirmed_tracks(self):
        return [t for t in self.tracks if t.confirmed and t.misses == 0]


# ---------------------------------------------------------------------------
class _D:
    def __init__(s, cx, cy, area): s.cx, s.cy, s.area = cx, cy, area


def _selftest():
    rng = np.random.default_rng(3)
    dt = 1 / 3.0
    # 1) moving target with noise and 25% dropouts + random clutter
    trk = TargetTracker()
    ids, confirmed_at = set(), None
    n = 60
    for k in range(n):
        tx, ty = 0.1 + 0.012 * k, 0.5 + 0.003 * k
        dets = []
        if rng.random() > 0.25:
            dets.append(_D(tx + rng.normal(0, 0.01), ty + rng.normal(0, 0.01),
                           1000 * (1 + rng.normal(0, 0.2))))
        if rng.random() < 0.3:
            dets.append(_D(rng.random(), rng.random(), 500))
        trk.step(dets, dt)
        for t in trk.confirmed_tracks():
            if abs(t.pos[0] - tx) < 0.1:
                ids.add(t.tid)
                confirmed_at = confirmed_at if confirmed_at is not None else k
    assert confirmed_at is not None, "never confirmed"
    assert confirmed_at <= 15, f"confirmed too late: {confirmed_at}"
    assert len(ids) == 1, f"ID switched: {ids}"
    main = [t for t in trk.tracks if t.confirmed]
    assert main, "no confirmed track at end"
    t = min(main, key=lambda t: abs(t.pos[0] - (0.1 + 0.012 * (n - 1))))
    ex, ey = 0.1 + 0.012 * (n - 1), 0.5 + 0.003 * (n - 1)
    assert abs(t.pos[0] - ex) < 0.03 and abs(t.pos[1] - ey) < 0.03, (t.pos, ex, ey)
    vx_true = 0.012 / dt
    assert abs(t.vel[0] - vx_true) < 0.03 and abs(t.vel[1] - 0.009 / dt) < 0.06, (t.vel, vx_true)

    # 2) clutter only: never confirms
    trk2 = TargetTracker()
    for k in range(300):
        dets = [_D(rng.random(), rng.random(), 500)] if rng.random() < 0.5 else []
        trk2.step(dets, dt)
        assert not any(t.confirmed for t in trk2.tracks), f"clutter confirmed at {k}"

    # 3) coasting: confirmed track survives a 5-frame blackout and is re-associated
    trk3 = TargetTracker()
    for k in range(15):
        trk3.step([_D(0.5 + 0.005 * k, 0.5, 800)], dt)
    tid = trk3.tracks[0].tid
    for k in range(5):
        trk3.step([], dt)
    assert trk3.tracks and trk3.tracks[0].tid == tid
    trk3.step([_D(0.5 + 0.005 * 20, 0.5, 800)], dt)
    assert trk3.tracks[0].tid == tid and trk3.tracks[0].misses == 0

    # 4) long blackout deletes the track
    for _ in range(MAX_MISSES + 2):
        trk3.step([], dt)
    assert not trk3.tracks

    # 5) flicker score ordering (diagnostic only)
    a = TargetTracker(); b = TargetTracker()
    for k in range(10):
        a.step([_D(0.5, 0.5, 1000 * (1 + 0.4 * rng.normal()))], dt)
        b.step([_D(0.5, 0.5, 1000)], dt)
    assert a.tracks[0].flicker > b.tracks[0].flicker + 0.1
    print(f"SELFTEST PASSED: confirmed at frame {confirmed_at}, single stable ID, "
          f"position/velocity within tolerance, clutter never confirms, coasting works, "
          f"long dropout deletes track.")


if __name__ == "__main__":
    _selftest()
