"""
Operator Context and Cabin State Estimator
Determines whether a detected worker is an active machinery cabin operator
or an external ground pedestrian, calculating G_i(t).
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
from safevision.config import OperatorContextConfig
from safevision.tracking.byte_track import TrackedObject


class OperatorState(Enum):
    PEDESTRIAN = "pedestrian"         # Ground worker exposed to moving machinery
    CONFIRMED_OPERATOR = "operator"   # Inside cabin, driving/operating the machine
    UNCERTAIN = "uncertain"           # Near cabin or partially occluded (conservative: not suppressed)


class OperatorContextEstimator:
    """
    Evaluates worker-machinery spatial containment and relative kinematics
    to determine cabin operator state and context multiplier G_i(t).

    FIX 1 - a worker is trusted to be the operator only if ALL of these hold:
      * their box sits inside the machine box (not just overlapping the cabin zone in 2D)
      * their box is much shorter than the machine (a seated driver, not a standing person)
      * they stay in the cabin region for several consecutive frames
      * they move WITH the machine (relative velocity small) - this check is never skipped,
        even when the machine is parked.
    Anything else is UNCERTAIN or PEDESTRIAN and keeps G = 1.0.
    """
    def __init__(self, config: Optional[OperatorContextConfig] = None, grace_frames: int = 15):
        self.config = config or OperatorContextConfig()
        self.grace_frames = grace_frames
        self._cabin_frames: Dict[int, int] = {}
        self._last_seen: Dict[int, int] = {}
        self._tick = 0

    def reset(self):
        self._cabin_frames.clear()
        self._last_seen.clear()
        self._tick = 0

    def start_frame(self, worker_ids: List[int]):
        """Call once per frame before evaluating workers; forgets workers gone for too long."""
        self._tick += 1
        for tid in worker_ids:
            self._last_seen[tid] = self._tick
        for tid in [t for t, seen in self._last_seen.items() if self._tick - seen > self.grace_frames]:
            self._last_seen.pop(tid, None)
            self._cabin_frames.pop(tid, None)

    def evaluate_operator_state(self, worker: TrackedObject,
                                machinery: List[TrackedObject]) -> Tuple[OperatorState, float, Optional[int]]:
        cfg = self.config
        w_box = worker.bbox
        w_area = max(1.0, (w_box[2] - w_box[0]) * (w_box[3] - w_box[1]))
        w_h = max(1.0, w_box[3] - w_box[1])

        def inter(a, b):
            return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))

        best_containment = 0.0
        best_machine_id = None
        best_ok = False        # geometry + size checks passed for the best machine
        best_rel_vel = float("inf")

        for m in machinery:
            m_box = m.bbox
            m_w = max(1.0, m_box[2] - m_box[0])
            m_h = max(1.0, m_box[3] - m_box[1])
            rel_y1, rel_x1, rel_y2, rel_x2 = cfg.cabin_relative_box
            cabin_box = np.array([m_box[0] + rel_x1 * m_w, m_box[1] + rel_y1 * m_h,
                                  m_box[0] + rel_x2 * m_w, m_box[1] + rel_y2 * m_h])

            containment = inter(w_box, cabin_box) / w_area
            if containment > best_containment:
                best_containment = containment
                best_machine_id = m.track_id
                inside_machine = inter(w_box, m_box) / w_area
                best_ok = (containment >= cfg.cabin_containment_threshold
                           and inside_machine >= cfg.min_inside_machine_ratio
                           and (w_h / m_h) <= cfg.max_worker_machine_height_ratio)
                best_rel_vel = float(np.linalg.norm(worker.velocity - m.velocity))

        tid = worker.track_id
        if best_ok:
            self._cabin_frames[tid] = self._cabin_frames.get(tid, 0) + 1
        else:
            self._cabin_frames[tid] = 0

        if best_ok and self._cabin_frames[tid] >= cfg.min_cabin_frames and best_rel_vel <= cfg.max_relative_velocity:
            state = OperatorState.CONFIRMED_OPERATOR
        elif best_containment >= 0.25:
            state = OperatorState.UNCERTAIN       # conservative: never auto-safe
        else:
            state = OperatorState.PEDESTRIAN

        multiplier = cfg.operator_multiplier if state == OperatorState.CONFIRMED_OPERATOR else cfg.pedestrian_multiplier
        return state, multiplier, best_machine_id
