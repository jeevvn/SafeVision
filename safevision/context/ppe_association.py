"""
Worker-to-PPE Geometric Association and Temporal Voting Engine
Assigns detected helmets and safety vests to tracked worker bounding boxes
based on anatomical region constraints and rolling temporal consensus.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from safevision.config import PPEAssociationConfig
from safevision.tracking.byte_track import TrackedObject


@dataclass
class WorkerPPEStatus:
    track_id: int
    has_helmet: bool = False
    has_vest: bool = False
    is_head_uncovered: bool = False
    ppe_penalty: float = 0.0          # P_i(t) in [0.0, 1.0]
    
    # Associated bounding boxes in current frame (if any)
    helmet_bbox: Optional[np.ndarray] = None
    vest_bbox: Optional[np.ndarray] = None
    
    # Association confidence scores
    helmet_score: float = 0.0
    vest_score: float = 0.0

    # Voting window history: deque of booleans
    helmet_history: deque = field(default_factory=lambda: deque(maxlen=7))
    vest_history: deque = field(default_factory=lambda: deque(maxlen=7))


def compute_intersection_area(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Computes area of intersection between box_a and box_b [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float((x2 - x1) * (y2 - y1))


class PPEAssociator:
    """
    Associates PPE detections with tracked worker instances using:
    1. Anatomical region partitioning (head/torso sub-regions).
    2. Geometric containment / IoU matching.
    3. Multi-frame sliding window temporal voting.
    """
    def __init__(self, config: Optional[PPEAssociationConfig] = None):
        self.config = config or PPEAssociationConfig()
        self.worker_ppe_states: Dict[int, WorkerPPEStatus] = {}
        self._last_seen: Dict[int, int] = {}
        self._tick = 0

    def reset(self):
        self.worker_ppe_states.clear()
        self._last_seen.clear()
        self._tick = 0

    def associate(self, workers: List[TrackedObject], ppe_detections: List[Dict]) -> Dict[int, WorkerPPEStatus]:
        """
        Args:
            workers: Active tracked worker objects from ByteTrack
            ppe_detections: List of dicts with keys 'bbox', 'class_name', 'score'
                           classes: 'helmet', 'vest', 'head_uncovered', etc.
        Returns:
            Dictionary mapping worker track_id -> WorkerPPEStatus
        """
        # FIX 3: forget a worker only after state_grace_frames of absence (not after a single missed frame)
        self._tick += 1
        for w in workers:
            self._last_seen[w.track_id] = self._tick
        stale_ids = [tid for tid, seen in self._last_seen.items()
                     if self._tick - seen > self.config.state_grace_frames]
        for tid in stale_ids:
            self._last_seen.pop(tid, None)
            self.worker_ppe_states.pop(tid, None)

        # Initialize status for any newly appeared workers
        for w in workers:
            if w.track_id not in self.worker_ppe_states:
                self.worker_ppe_states[w.track_id] = WorkerPPEStatus(
                    track_id=w.track_id,
                    helmet_history=deque(maxlen=self.config.voting_window_frames),
                    vest_history=deque(maxlen=self.config.voting_window_frames)
                )

        # Categorize PPE detections
        helmets = [d for d in ppe_detections if d['class_name'] in ('helmet', 'hat')]
        vests = [d for d in ppe_detections if d['class_name'] in ('vest', 'safety_vest')]
        uncovered = [d for d in ppe_detections if d['class_name'] in ('head_uncovered', 'no_helmet', 'head')]

        # Frame-level assignment candidates
        frame_helmet_assigned: Dict[int, Tuple[Dict, float]] = {}  # worker_id -> (det, score)
        frame_vest_assigned: Dict[int, Tuple[Dict, float]] = {}
        frame_uncovered_assigned: Dict[int, Tuple[Dict, float]] = {}

        # 1. Match Helmets to Workers (Head region)
        for h_det in helmets:
            h_box = np.array(h_det['bbox'], dtype=float)
            h_area = max(1.0, (h_box[2] - h_box[0]) * (h_box[3] - h_box[1]))
            best_worker_id = None
            best_agreement = 0.0

            for w in workers:
                w_box = w.bbox
                w_h = max(1.0, w_box[3] - w_box[1])
                # Head region: top of worker down to head_interval
                head_region = np.array([
                    w_box[0] - 0.1 * (w_box[2] - w_box[0]),  # slight horizontal margin
                    w_box[1],
                    w_box[2] + 0.1 * (w_box[2] - w_box[0]),
                    w_box[1] + self.config.head_region_interval[1] * w_h
                ])

                inter_area = compute_intersection_area(h_box, head_region)
                containment = inter_area / h_area
                if containment >= self.config.min_association_containment and containment > best_agreement:
                    best_agreement = containment
                    best_worker_id = w.track_id

            if best_worker_id is not None:
                if (best_worker_id not in frame_helmet_assigned or 
                    best_agreement > frame_helmet_assigned[best_worker_id][1]):
                    frame_helmet_assigned[best_worker_id] = (h_det, best_agreement)

        # 2. Match Vests to Workers (Torso region)
        for v_det in vests:
            v_box = np.array(v_det['bbox'], dtype=float)
            v_area = max(1.0, (v_box[2] - v_box[0]) * (v_box[3] - v_box[1]))
            best_worker_id = None
            best_agreement = 0.0

            for w in workers:
                w_box = w.bbox
                w_h = max(1.0, w_box[3] - w_box[1])
                torso_region = np.array([
                    w_box[0] - 0.1 * (w_box[2] - w_box[0]),
                    w_box[1] + self.config.torso_region_interval[0] * w_h,
                    w_box[2] + 0.1 * (w_box[2] - w_box[0]),
                    w_box[1] + self.config.torso_region_interval[1] * w_h
                ])

                inter_area = compute_intersection_area(v_box, torso_region)
                containment = inter_area / v_area
                if containment >= self.config.min_association_containment and containment > best_agreement:
                    best_agreement = containment
                    best_worker_id = w.track_id

            if best_worker_id is not None:
                if (best_worker_id not in frame_vest_assigned or 
                    best_agreement > frame_vest_assigned[best_worker_id][1]):
                    frame_vest_assigned[best_worker_id] = (v_det, best_agreement)

        # 3. Match Uncovered Head evidence (if available in ontology)
        for u_det in uncovered:
            u_box = np.array(u_det['bbox'], dtype=float)
            u_area = max(1.0, (u_box[2] - u_box[0]) * (u_box[3] - u_box[1]))
            for w in workers:
                w_box = w.bbox
                w_h = max(1.0, w_box[3] - w_box[1])
                head_region = np.array([
                    w_box[0], w_box[1], w_box[2],
                    w_box[1] + self.config.head_region_interval[1] * w_h
                ])
                inter = compute_intersection_area(u_box, head_region)
                if inter / u_area >= self.config.min_association_containment:
                    frame_uncovered_assigned[w.track_id] = (u_det, inter / u_area)

        # 4. Temporal voting & consensus update for each active worker
        results: Dict[int, WorkerPPEStatus] = {}

        for w in workers:
            tid = w.track_id
            status = self.worker_ppe_states[tid]

            # Current frame observations
            curr_has_helmet = (tid in frame_helmet_assigned)
            curr_has_vest = (tid in frame_vest_assigned)
            curr_uncovered = (tid in frame_uncovered_assigned)

            # If explicit uncovered head detected, overrides doubtful helmet
            if curr_uncovered and not curr_has_helmet:
                status.is_head_uncovered = True
            else:
                status.is_head_uncovered = False

            # Append to rolling temporal history
            status.helmet_history.append(curr_has_helmet)
            status.vest_history.append(curr_has_vest)

            # Temporal voting: consensus ratio across window
            helmet_vote = (sum(status.helmet_history) / len(status.helmet_history)) if status.helmet_history else 0.0
            vest_vote = (sum(status.vest_history) / len(status.vest_history)) if status.vest_history else 0.0

            status.has_helmet = (helmet_vote >= self.config.voting_positive_threshold)
            status.has_vest = (vest_vote >= self.config.voting_positive_threshold)

            # Bounding box references for visual overlays
            if curr_has_helmet:
                status.helmet_bbox = np.array(frame_helmet_assigned[tid][0]['bbox'])
                status.helmet_score = frame_helmet_assigned[tid][0]['score']
            else:
                status.helmet_bbox = None
                status.helmet_score = 0.0

            if curr_has_vest:
                status.vest_bbox = np.array(frame_vest_assigned[tid][0]['bbox'])
                status.vest_score = frame_vest_assigned[tid][0]['score']
            else:
                status.vest_bbox = None
                status.vest_score = 0.0

            # Calculate PPE non-compliance penalty P_i(t) in [0.0, 1.0]
            penalty = 0.0
            if not status.has_helmet:
                penalty += self.config.penalty_missing_helmet
            if not status.has_vest:
                penalty += self.config.penalty_missing_vest

            status.ppe_penalty = float(np.clip(penalty, 0.0, 1.0))
            results[tid] = status

        return results
