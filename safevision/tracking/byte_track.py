"""
ByteTrack Multi-Object Tracker Implementation (Pure NumPy & SciPy)
Implements Kalman Filter state estimation, two-stage confidence matching,
and velocity vector tracking for Workers and Machinery.
"""

from collections import deque
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment


class TrackState(Enum):
    TENTATIVE = 1
    CONFIRMED = 2
    LOST = 3
    REMOVED = 4


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Computes pairwise IoU between two sets of bounding boxes [x1, y1, x2, y2]."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.empty((len(boxes_a), len(boxes_b)), dtype=float)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    iw = np.maximum(0.0, np.minimum(boxes_a[:, 2:3], boxes_b[:, 2]) - np.maximum(boxes_a[:, 0:1], boxes_b[:, 0]))
    ih = np.maximum(0.0, np.minimum(boxes_a[:, 3:4], boxes_b[:, 3]) - np.maximum(boxes_a[:, 1:2], boxes_b[:, 1]))
    intersection = iw * ih

    union = area_a[:, None] + area_b[None, :] - intersection
    return np.clip(intersection / np.maximum(union, 1e-6), 0.0, 1.0)


class KalmanBoxTracker:
    """
    Kalman Filter for single object bounding box [x1, y1, x2, y2].
    State vector: [cx, cy, s, r, vx, vy, vs], where s=area, r=aspect_ratio (w/h).
    """
    # FIX 6: no class-level ID counter. Each ByteTracker owns its own counter and passes the id in.
    VELOCITY_WINDOW = 6        # frames of position history used for the velocity fit
    VELOCITY_DEADBAND = 0.8    # px/frame below which a box is treated as stationary

    def __init__(self, bbox: np.ndarray, class_name: str, score: float, track_id: int):
        self.class_name = class_name
        self.score = score
        self.id = track_id

        # State: [cx, cy, s, r, vx, vy, vs]
        w = max(1.0, bbox[2] - bbox[0])
        h = max(1.0, bbox[3] - bbox[1])
        cx = bbox[0] + w / 2.0
        cy = bbox[1] + h / 2.0
        s = w * h
        r = w / h

        self.x = np.array([cx, cy, s, r, 0.0, 0.0, 0.0], dtype=float)

        # Covariance matrices
        self.P = np.diag([10.0, 10.0, 50.0, 1.0, 1000.0, 1000.0, 1000.0])
        self.Q = np.diag([1.0, 1.0, 1.0, 0.01, 10.0, 10.0, 10.0])
        self.R = np.diag([1.0, 1.0, 10.0, 0.1])

        # State transition matrix F
        self.F = np.eye(7)
        self.F[0, 4] = 1.0  # cx += vx
        self.F[1, 5] = 1.0  # cy += vy
        self.F[2, 6] = 1.0  # s += vs

        # Measurement matrix H
        self.H = np.zeros((4, 7))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.state = TrackState.TENTATIVE
        self.history: List[np.ndarray] = [bbox.copy()]
        self.velocity: np.ndarray = np.array([0.0, 0.0])  # [vx, vy]
        self._pos_hist = deque([(0, cx, cy)], maxlen=self.VELOCITY_WINDOW)

    def _windowed_velocity(self) -> np.ndarray:
        """FIX 4: least-squares slope over recent observed centres, EMA-smoothed, with a dead-band.
        The raw Kalman velocity state reacts to every pixel of detector jitter."""
        if len(self._pos_hist) < 3:
            return np.zeros(2)
        t = np.array([p[0] for p in self._pos_hist], dtype=float)
        if t[-1] - t[0] < 2:
            return np.zeros(2)
        vx = np.polyfit(t, [p[1] for p in self._pos_hist], 1)[0]
        vy = np.polyfit(t, [p[2] for p in self._pos_hist], 1)[0]
        v = 0.6 * np.array([vx, vy]) + 0.4 * self.velocity
        if np.linalg.norm(v) < self.VELOCITY_DEADBAND:
            return np.zeros(2)
        return v

    def predict(self) -> np.ndarray:
        """Advances state vector and returns predicted bounding box [x1, y1, x2, y2]."""
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        self.age += 1
        self.time_since_update += 1
        return self.get_bbox()

    def update(self, bbox: np.ndarray, score: float):
        """Updates Kalman Filter with observed bounding box measurement."""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.score = score

        w = max(1.0, bbox[2] - bbox[0])
        h = max(1.0, bbox[3] - bbox[1])
        cx = bbox[0] + w / 2.0
        cy = bbox[1] + h / 2.0
        s = w * h
        r = w / h
        z = np.array([cx, cy, s, r])

        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        self.x += np.dot(K, y)
        self.P = np.dot(np.eye(7) - np.dot(K, self.H), self.P)

        # FIX 4: windowed, smoothed velocity instead of the raw (noisy) Kalman velocity state
        self._pos_hist.append((self.age, cx, cy))
        self.velocity = self._windowed_velocity()
        current_bbox = self.get_bbox()
        self.history.append(current_bbox)
        if len(self.history) > 60:
            self.history.pop(0)

    def get_bbox(self) -> np.ndarray:
        """Converts state [cx, cy, s, r] to bounding box [x1, y1, x2, y2]."""
        cx, cy, s, r = self.x[0], self.x[1], max(1e-2, self.x[2]), max(1e-2, self.x[3])
        w = np.sqrt(s * r)
        h = s / max(1e-2, w)
        return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0])


class TrackedObject:
    """Public interface for a tracked object."""
    def __init__(self, track_id: int, bbox: np.ndarray, class_name: str,
                 score: float, velocity: np.ndarray, is_confirmed: bool):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.class_name = class_name
        self.score = score
        self.velocity = velocity  # [vx, vy]
        self.speed = float(np.linalg.norm(velocity))
        self.is_confirmed = is_confirmed

    @property
    def center(self) -> Tuple[float, float]:
        return float((self.bbox[0] + self.bbox[2]) / 2.0), float((self.bbox[1] + self.bbox[3]) / 2.0)

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])


class ByteTracker:
    """
    ByteTrack implementation for multi-class tracking.
    Separates high-confidence and low-confidence detections to preserve tracks
    through occlusion.
    """
    def __init__(self, high_thresh: float = 0.5, low_thresh: float = 0.2,
                 match_thresh: float = 0.7, max_lost_frames: int = 25,
                 min_hits_to_confirm: int = 2):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.max_lost_frames = max_lost_frames
        self.min_hits_to_confirm = min_hits_to_confirm
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0
        self._next_id = 1   # FIX 6: per-tracker counter

    def reset(self):
        self.trackers.clear()
        self.frame_count = 0
        self._next_id = 1

    def update(self, detections: List[Dict]) -> List[TrackedObject]:
        """
        Detections format: list of dicts with:
        {'bbox': [x1, y1, x2, y2], 'score': float, 'class_name': str}
        """
        self.frame_count += 1

        # 1. Predict all current tracks
        predicted_boxes = []
        for t in self.trackers:
            pred_box = t.predict()
            predicted_boxes.append(pred_box)

        # 2. Split detections into high confidence and low confidence
        high_dets = [d for d in detections if d['score'] >= self.high_thresh]
        low_dets = [d for d in detections if self.low_thresh <= d['score'] < self.high_thresh]

        unmatched_track_indices = list(range(len(self.trackers)))
        unmatched_high_indices = list(range(len(high_dets)))

        # 3. First association: match high confidence detections with all tracks
        if len(self.trackers) > 0 and len(high_dets) > 0:
            track_boxes = np.array([t.get_bbox() for t in self.trackers])
            high_boxes = np.array([d['bbox'] for d in high_dets])
            iou_mat = compute_iou_matrix(track_boxes, high_boxes)

            # Penalize cross-class matches
            for i, t in enumerate(self.trackers):
                for j, d in enumerate(high_dets):
                    # Only match same general category (worker matches worker, machinery matches machinery)
                    is_t_worker = (t.class_name == 'worker' or t.class_name == 'person')
                    is_d_worker = (d['class_name'] == 'worker' or d['class_name'] == 'person')
                    if is_t_worker != is_d_worker:
                        iou_mat[i, j] = 0.0

            cost_mat = 1.0 - iou_mat
            row_ind, col_ind = linear_sum_assignment(cost_mat)

            matched_track_set = set()
            matched_high_set = set()
            for r, c in zip(row_ind, col_ind):
                if cost_mat[r, c] < (1.0 - (1.0 - self.match_thresh)):  # IoU >= 1 - match_thresh
                    self.trackers[r].update(high_dets[c]['bbox'], high_dets[c]['score'])
                    matched_track_set.add(r)
                    matched_high_set.add(c)

            unmatched_track_indices = [i for i in range(len(self.trackers)) if i not in matched_track_set]
            unmatched_high_indices = [j for j in range(len(high_dets)) if j not in matched_high_set]

        # 4. Second association: match remaining tracks with low-confidence detections
        if len(unmatched_track_indices) > 0 and len(low_dets) > 0:
            remaining_track_boxes = np.array([self.trackers[i].get_bbox() for i in unmatched_track_indices])
            low_boxes = np.array([d['bbox'] for d in low_dets])
            iou_mat_low = compute_iou_matrix(remaining_track_boxes, low_boxes)

            cost_mat_low = 1.0 - iou_mat_low
            row_ind_low, col_ind_low = linear_sum_assignment(cost_mat_low)

            matched_second_tracks = set()
            for r, c in zip(row_ind_low, col_ind_low):
                if cost_mat_low[r, c] < 0.6:  # IoU >= 0.4
                    orig_track_idx = unmatched_track_indices[r]
                    self.trackers[orig_track_idx].update(low_dets[c]['bbox'], low_dets[c]['score'])
                    matched_second_tracks.add(orig_track_idx)

            unmatched_track_indices = [i for i in unmatched_track_indices if i not in matched_second_tracks]

        # 5. Initialize new tracks from unmatched high-confidence detections
        for c in unmatched_high_indices:
            det = high_dets[c]
            new_tracker = KalmanBoxTracker(
                bbox=np.array(det['bbox']),
                class_name=det['class_name'],
                score=det['score'],
                track_id=self._next_id
            )
            self._next_id += 1
            self.trackers.append(new_tracker)

        # 6. Update track states and purge dead tracks
        active_outputs: List[TrackedObject] = []
        surviving_trackers: List[KalmanBoxTracker] = []

        for t in self.trackers:
            if t.hits >= self.min_hits_to_confirm:
                t.state = TrackState.CONFIRMED

            if t.time_since_update > self.max_lost_frames:
                t.state = TrackState.REMOVED
            else:
                surviving_trackers.append(t)

            if t.time_since_update == 0 and t.state == TrackState.CONFIRMED:
                active_outputs.append(TrackedObject(
                    track_id=t.id,
                    bbox=t.get_bbox(),
                    class_name=t.class_name,
                    score=t.score,
                    velocity=t.velocity,
                    is_confirmed=(t.state == TrackState.CONFIRMED)
                ))

        self.trackers = surviving_trackers
        return active_outputs
