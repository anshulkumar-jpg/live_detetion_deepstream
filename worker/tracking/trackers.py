"""Multi-object trackers in pure NumPy: ByteTrack, OC-SORT, BoT-SORT, StrongSORT.

Every tracker has the same interface (one instance per camera):

    ids = tracker.update(boxes_xyxy[N,4], scores[N], classes[N], feats=None | embeddings[N,D])

and returns an int track id per detection (-1 when a detection was not attached to any track).

Fidelity notes (what is and is not in each implementation):
* ByteTrack  - two-stage association (high then low score boxes), Kalman (x, y, aspect, height), lost-track buffer.
* OC-SORT    - observation-centric momentum (direction consistency), observation-centric recovery
               (re-match on last observation) and observation-centric re-update (virtual trajectory).
* BoT-SORT   - ByteTrack association with a Kalman filter on (x, y, w, h) and IoU/appearance fusion.
               Appearance is used only when Re-ID embeddings are supplied; camera-motion compensation is
               omitted (CCTV cameras are static).
* StrongSORT - DeepSORT-style matching cascade, EMA appearance gallery and NSA Kalman (noise scaled by
               detection confidence). Appearance is used only when embeddings are supplied. Mahalanobis gating
               is replaced by IoU gating.
Without embeddings BoT-SORT and StrongSORT run motion-only.
"""
from __future__ import annotations

import numpy as np

try:  # SciPy gives the optimal assignment; a greedy fallback keeps the tracker usable without it
    from scipy.optimize import linear_sum_assignment as _lsa_scipy
except Exception:  # pragma: no cover
    _lsa_scipy = None


# ------------------------------------------------------------------ helpers
def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between xyxy boxes a[N,4] and b[M,4] -> [N,M]."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def _assign(cost: np.ndarray) -> list[tuple[int, int]]:
    if cost.size == 0:
        return []
    if _lsa_scipy is not None:
        r, c = _lsa_scipy(cost)
        return list(zip(r.tolist(), c.tolist()))
    pairs, used_r, used_c = [], set(), set()
    for idx in np.argsort(cost, axis=None):
        i, j = divmod(int(idx), cost.shape[1])
        if i not in used_r and j not in used_c:
            used_r.add(i)
            used_c.add(j)
            pairs.append((i, j))
    return pairs


def match(cost: np.ndarray, thresh: float):
    """Assignment that rejects pairs with cost > thresh -> (matches, unmatched_rows, unmatched_cols)."""
    n, m = cost.shape
    if n == 0 or m == 0:
        return [], list(range(n)), list(range(m))
    matches, ur, uc = [], set(range(n)), set(range(m))
    for i, j in _assign(cost):
        if cost[i, j] <= thresh:
            matches.append((i, j))
            ur.discard(i)
            uc.discard(j)
    return matches, sorted(ur), sorted(uc)


def cosine_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-9)
    return 1.0 - a @ b.T


def xyxy_to_xyah(b):
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w / max(h, 1e-6), h])


def xyxy_to_xywh(b):
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w, h])


# ------------------------------------------------------------------ Kalman (ByteTrack / BoT-SORT / StrongSORT)
class Kalman:
    """Constant-velocity Kalman filter over an 8-d state [x y a h vx vy va vh] ('xyah') or [x y w h ...] ('xywh')."""

    def __init__(self, mode: str = "xyah") -> None:
        self.mode = mode
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, 4 + i] = 1.0
        self.H = np.eye(4, 8)
        self.wp, self.wv = 1.0 / 20, 1.0 / 160

    def _scale(self, mean):  # per-component scale of the noise
        if self.mode == "xyah":
            return np.array([mean[3], mean[3], 1.0, mean[3]])
        return np.array([mean[2], mean[3], mean[2], mean[3]])

    def initiate(self, z):
        mean = np.r_[z, np.zeros(4)]
        s = self._scale(z)
        std = np.r_[2 * self.wp * s, 10 * self.wv * s]
        if self.mode == "xyah":
            std[2], std[6] = 1e-2, 1e-5
        return mean, np.diag(np.square(std))

    def predict(self, mean, cov):
        s = self._scale(mean)
        std_p, std_v = self.wp * s, self.wv * s
        if self.mode == "xyah":
            std_p[2], std_v[2] = 1e-2, 1e-5
        q = np.diag(np.square(np.r_[std_p, std_v]))
        return self.F @ mean, self.F @ cov @ self.F.T + q

    def update(self, mean, cov, z, conf: float | None = None):
        s = self._scale(mean)
        std = self.wp * s
        if self.mode == "xyah":
            std[2] = 1e-1
        r = np.diag(np.square(std))
        if conf is not None:  # NSA: the more confident the detection, the smaller the measurement noise
            r = r * max(1.0 - conf, 1e-2)
        pm = self.H @ mean
        pc = self.H @ cov @ self.H.T + r
        gain = np.linalg.solve(pc, self.H @ cov.T).T
        return mean + gain @ (z - pm), cov - gain @ pc @ gain.T

    def to_xyxy(self, mean):
        if self.mode == "xyah":
            h = mean[3]
            w = mean[2] * h
        else:
            w, h = mean[2], mean[3]
        return np.array([mean[0] - w / 2, mean[1] - h / 2, mean[0] + w / 2, mean[1] + h / 2])

    def measure(self, box):
        return xyxy_to_xyah(box) if self.mode == "xyah" else xyxy_to_xywh(box)


class _T:
    """Track used by ByteTrack / BoT-SORT / StrongSORT."""

    def __init__(self, tid, box, score, cls, kf: Kalman, feat=None):
        self.id, self.cls, self.score, self.kf = tid, cls, score, kf
        self.mean, self.cov = kf.initiate(kf.measure(box))
        self.lost = 0            # frames since last matched
        self.hits = 1
        self.age = 1
        self.feat = None if feat is None else feat / max(np.linalg.norm(feat), 1e-9)

    def predict(self):
        self.mean, self.cov = self.kf.predict(self.mean, self.cov)
        self.age += 1
        self.lost += 1

    @property
    def box(self):
        return self.kf.to_xyxy(self.mean)

    def update(self, box, score, feat=None, nsa=False, alpha=0.9):
        self.mean, self.cov = self.kf.update(self.mean, self.cov, self.kf.measure(box), score if nsa else None)
        self.score, self.lost = score, 0
        self.hits += 1
        if feat is not None:
            f = feat / max(np.linalg.norm(feat), 1e-9)
            self.feat = f if self.feat is None else alpha * self.feat + (1 - alpha) * f
            self.feat = self.feat / max(np.linalg.norm(self.feat), 1e-9)


def _motion_cost(tracks, boxes, classes):
    tb = np.array([t.box for t in tracks]).reshape(-1, 4)
    cost = 1.0 - iou_matrix(tb, boxes)
    for i, t in enumerate(tracks):  # never match across classes
        cost[i, np.asarray(classes) != t.cls] = 1.0
    return cost


# ------------------------------------------------------------------ ByteTrack / BoT-SORT
class ByteTrack:
    name = "ByteTrack"

    def __init__(self, track_thresh=0.5, low_thresh=0.1, match_thresh=0.8, second_thresh=0.5, new_thresh=0.6,
                 buffer=30, mode="xyah", appearance=False, prox=0.5, app=0.25):
        self.track_thresh, self.low_thresh = track_thresh, low_thresh
        self.match_thresh, self.second_thresh, self.new_thresh = match_thresh, second_thresh, new_thresh
        self.buffer, self.appearance, self.prox, self.app = buffer, appearance, prox, app
        self.kf = Kalman(mode)
        self.tracks: list[_T] = []
        self.next_id = 1

    def _cost(self, tracks, boxes, classes, feats):
        cost = _motion_cost(tracks, boxes, classes)
        if self.appearance and feats is not None and len(tracks) and len(boxes):
            tf = [t.feat for t in tracks]
            if all(f is not None for f in tf):
                emb = cosine_dist(np.array(tf), feats) / 2.0
                emb[emb > self.app] = 1.0
                emb[cost > self.prox] = 1.0
                cost = np.minimum(cost, emb)
        return cost

    def update(self, boxes, scores, classes, feats=None):
        boxes = np.asarray(boxes, float).reshape(-1, 4)
        scores, classes = np.asarray(scores, float), np.asarray(classes)
        n = len(boxes)
        ids = -np.ones(n, int)
        for t in self.tracks:
            t.predict()
        high = [i for i in range(n) if scores[i] >= self.track_thresh]
        low = [i for i in range(n) if self.low_thresh <= scores[i] < self.track_thresh]
        fh = feats[high] if feats is not None and high else None

        # 1) high-score detections vs all tracks (tracked + recently lost)
        pool = list(self.tracks)
        m, ut, ud = match(self._cost(pool, boxes[high], classes[high], fh), self.match_thresh)
        for ti, di in m:
            pool[ti].update(boxes[high[di]], scores[high[di]], None if fh is None else fh[di])
            ids[high[di]] = pool[ti].id
        # 2) low-score detections vs tracks that were being tracked until the previous frame
        rest = [pool[i] for i in ut if pool[i].lost == 1]
        m2, ut2, _ = match(_motion_cost(rest, boxes[low], classes[low]) if rest and low else np.zeros((len(rest), len(low))),
                           self.second_thresh)
        for ti, di in m2:
            rest[ti].update(boxes[low[di]], scores[low[di]])
            ids[low[di]] = rest[ti].id
        # 3) new tracks from unmatched high-score detections
        for di in ud:
            i = high[di]
            if scores[i] >= self.new_thresh:
                t = _T(self.next_id, boxes[i], scores[i], classes[i], self.kf, None if fh is None else fh[di])
                self.next_id += 1
                self.tracks.append(t)
                ids[i] = t.id
        self.tracks = [t for t in self.tracks if t.lost <= self.buffer]
        return ids


class BoTSORT(ByteTrack):
    name = "BoTSORT"

    def __init__(self, **kw):
        super().__init__(track_thresh=0.6, new_thresh=0.7, match_thresh=0.8, mode="xywh", appearance=True, **kw)


# ------------------------------------------------------------------ OC-SORT
class _Kalman7:
    """SORT Kalman filter on [x y s r vx vy vs] with OC-SORT's freeze / re-update (ORU)."""

    def __init__(self, z):
        self.F = np.array([[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]], float)
        self.H = np.eye(4, 7)
        self.R = np.eye(4)
        self.R[2:, 2:] *= 10.0
        self.P = np.eye(7)
        self.P[4:, 4:] *= 1000.0
        self.P *= 10.0
        self.Q = np.eye(7)
        self.Q[-1, -1] *= 0.01
        self.Q[4:, 4:] *= 0.01
        self.x = np.zeros(7)
        self.x[:4] = z
        self.saved = None

    def predict(self):
        if self.x[6] + self.x[2] <= 0:
            self.x[6] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        y = z - self.H @ self.x
        s = self.H @ self.P @ self.H.T + self.R
        k = np.linalg.solve(s, self.H @ self.P.T).T
        self.x = self.x + k @ y
        self.P = (np.eye(7) - k @ self.H) @ self.P

    def box(self):
        s, r = max(self.x[2], 1e-6), max(self.x[3], 1e-6)
        w = np.sqrt(s * r)
        h = s / w
        return np.array([self.x[0] - w / 2, self.x[1] - h / 2, self.x[0] + w / 2, self.x[1] + h / 2])


def _xyxy_to_z(b):
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w * h, w / max(h, 1e-6)])


def _z_to_xyxy(z):
    w = np.sqrt(max(z[2] * z[3], 1e-6))
    h = max(z[2], 1e-6) / w
    return np.array([z[0] - w / 2, z[1] - h / 2, z[0] + w / 2, z[1] + h / 2])


def _direction(b1, b2):
    c1 = np.array([(b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2])
    c2 = np.array([(b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2])
    d = c2 - c1
    return d / (np.linalg.norm(d) + 1e-6)


class _OT:
    def __init__(self, tid, box, score, cls, delta_t=3):
        self.id, self.cls, self.score, self.delta_t = tid, cls, score, delta_t
        self.kf = _Kalman7(_xyxy_to_z(box))
        self.age = 0
        self.tsu = 0          # time since update
        self.hits = 1
        self.streak = 1
        self.last_obs = box.copy()
        self.obs: dict[int, np.ndarray] = {self.age: box.copy()}
        self.velocity = None
        self.saved = (self.kf.x.copy(), self.kf.P.copy())
        self.last_z = _xyxy_to_z(box)

    def prev_obs(self):
        for i in range(self.delta_t):
            dt = self.delta_t - i
            if self.age - dt in self.obs:
                return self.obs[self.age - dt]
        return self.last_obs

    def predict(self):
        self.kf.predict()
        self.age += 1
        if self.tsu > 0:
            self.streak = 0
        self.tsu += 1

    def update(self, box, score):
        prev = self.prev_obs()
        self.velocity = _direction(prev, box)
        z = _xyxy_to_z(box)
        if self.tsu > 1:  # observation-centric re-update: replay a virtual trajectory from the last observation
            gap = self.tsu
            self.kf.x, self.kf.P = self.saved[0].copy(), self.saved[1].copy()
            for i in range(1, gap + 1):
                self.kf.predict()
                self.kf.update(self.last_z + (z - self.last_z) * (i / gap))
        else:
            self.kf.update(z)
        self.last_obs, self.last_z, self.score = box.copy(), z, score
        self.obs[self.age] = box.copy()
        self.saved = (self.kf.x.copy(), self.kf.P.copy())
        self.tsu = 0
        self.hits += 1
        self.streak += 1


class OCSORT:
    name = "OCSORT"

    def __init__(self, det_thresh=0.3, max_age=30, iou_thresh=0.3, delta_t=3, vdc_weight=0.2):
        self.det_thresh, self.max_age, self.iou_thresh = det_thresh, max_age, iou_thresh
        self.delta_t, self.vdc = delta_t, vdc_weight
        self.tracks: list[_OT] = []
        self.next_id = 1

    def update(self, boxes, scores, classes, feats=None):
        boxes = np.asarray(boxes, float).reshape(-1, 4)
        scores, classes = np.asarray(scores, float), np.asarray(classes)
        n = len(boxes)
        ids = -np.ones(n, int)
        keep = [i for i in range(n) if scores[i] >= self.det_thresh]
        for t in self.tracks:
            t.predict()
        tb = np.array([t.kf.box() for t in self.tracks]).reshape(-1, 4)
        db = boxes[keep]
        ious = iou_matrix(tb, db)                                        # [tracks, dets]
        for i, t in enumerate(self.tracks):
            ious[i, classes[keep] != t.cls] = 0.0
        # observation-centric momentum: reward detections lying in the direction the track was moving
        angle = np.zeros_like(ious)
        for i, t in enumerate(self.tracks):
            if t.velocity is None:
                continue
            for j in range(len(db)):
                d = _direction(t.prev_obs(), db[j])
                cos = float(np.clip(t.velocity @ d, -1, 1))
                angle[i, j] = (np.pi / 2 - abs(np.arccos(cos))) / np.pi * self.vdc * scores[keep[j]]
        cost = -(ious + angle)
        matched_t, matched_d = set(), set()
        if len(self.tracks) and len(db):
            for ti, di in _assign(cost):
                if ious[ti, di] >= self.iou_thresh:
                    self.tracks[ti].update(db[di], scores[keep[di]])
                    ids[keep[di]] = self.tracks[ti].id
                    matched_t.add(ti)
                    matched_d.add(di)
        # observation-centric recovery: match leftovers to the track's last observation
        ut = [i for i in range(len(self.tracks)) if i not in matched_t]
        ud = [j for j in range(len(db)) if j not in matched_d]
        if ut and ud:
            lo = np.array([self.tracks[i].last_obs for i in ut])
            iou_left = iou_matrix(lo, db[ud])
            for a, i in enumerate(ut):
                iou_left[a, classes[keep][ud] != self.tracks[i].cls] = 0.0
            for a, b in _assign(-iou_left):
                if iou_left[a, b] >= self.iou_thresh:
                    self.tracks[ut[a]].update(db[ud[b]], scores[keep[ud[b]]])
                    ids[keep[ud[b]]] = self.tracks[ut[a]].id
                    matched_d.add(ud[b])
                    matched_t.add(ut[a])
        for j in range(len(db)):
            if j not in matched_d and ids[keep[j]] < 0:
                t = _OT(self.next_id, db[j], scores[keep[j]], classes[keep[j]], self.delta_t)
                self.next_id += 1
                self.tracks.append(t)
                ids[keep[j]] = t.id
        self.tracks = [t for t in self.tracks if t.tsu <= self.max_age]
        return ids


# ------------------------------------------------------------------ StrongSORT
class StrongSORT:
    name = "StrongSORT"

    def __init__(self, max_age=30, n_init=3, max_iou_dist=0.7, lam=0.98, app_gate=0.3, ema=0.9):
        self.max_age, self.n_init, self.max_iou_dist = max_age, n_init, max_iou_dist
        self.lam, self.app_gate, self.ema = lam, app_gate, ema
        self.kf = Kalman("xyah")
        self.tracks: list[_T] = []
        self.next_id = 1

    @staticmethod
    def _confirmed(t):
        return t.hits >= 3

    def _cost(self, tracks, boxes, classes, feats):
        mot = _motion_cost(tracks, boxes, classes)
        if feats is not None and len(tracks) and len(boxes) and all(t.feat is not None for t in tracks):
            app = cosine_dist(np.array([t.feat for t in tracks]), feats)
            cost = self.lam * app + (1 - self.lam) * mot
            cost[mot > self.max_iou_dist] = 1e5            # motion gate
            cost[app > self.app_gate * 2] = 1e5            # appearance gate
            return cost
        mot[mot > self.max_iou_dist] = 1e5
        return mot

    def update(self, boxes, scores, classes, feats=None):
        boxes = np.asarray(boxes, float).reshape(-1, 4)
        scores, classes = np.asarray(scores, float), np.asarray(classes)
        n = len(boxes)
        ids = -np.ones(n, int)
        for t in self.tracks:
            t.predict()
        unmatched = list(range(n))
        matched: set[int] = set()
        # matching cascade over confirmed tracks, most recently seen first
        confirmed = [t for t in self.tracks if self._confirmed(t)]
        for level in range(1, self.max_age + 1):
            if not unmatched:
                break
            sub = [t for t in confirmed if t.lost == level]
            if not sub:
                continue
            fd = feats[unmatched] if feats is not None else None
            m, _, ud = match(self._cost(sub, boxes[unmatched], classes[unmatched], fd), 1e4)
            for ti, di in m:
                i = unmatched[di]
                sub[ti].update(boxes[i], scores[i], None if feats is None else feats[i], nsa=True, alpha=self.ema)
                ids[i] = sub[ti].id
                matched.add(sub[ti].id)
            unmatched = [unmatched[d] for d in ud]
        # IoU matching for tentative tracks and for confirmed tracks that were seen on the previous frame
        cand = [t for t in self.tracks if t.id not in matched and t.lost == 1]
        if cand and unmatched:
            cost = _motion_cost(cand, boxes[unmatched], classes[unmatched])
            m, _, ud = match(cost, self.max_iou_dist)
            for ti, di in m:
                i = unmatched[di]
                cand[ti].update(boxes[i], scores[i], None if feats is None else feats[i], nsa=True, alpha=self.ema)
                ids[i] = cand[ti].id
            unmatched = [unmatched[d] for d in ud]
        for i in unmatched:
            t = _T(self.next_id, boxes[i], scores[i], classes[i], self.kf, None if feats is None else feats[i])
            self.next_id += 1
            self.tracks.append(t)
            ids[i] = t.id
        # tentative tracks that are lost are dropped immediately, confirmed ones after max_age frames
        self.tracks = [t for t in self.tracks if (t.lost <= self.max_age if self._confirmed(t) else t.lost <= 1)]
        return ids


_FACTORY = {"ByteTrack": ByteTrack, "OCSORT": OCSORT, "BoTSORT": BoTSORT, "StrongSORT": StrongSORT}


def make_tracker(name: str):
    try:
        return _FACTORY[name]()
    except KeyError:
        raise ValueError(f"unknown python tracker '{name}' (choose from {sorted(_FACTORY)})") from None
