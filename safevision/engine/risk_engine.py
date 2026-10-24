"""
Operator-Aware Adaptive Temporal Risk Engine
Computes worker-specific risk R_i(t) using danger zone occupancy,
machinery motion/approach, exposure duration, and PPE compliance,
governed by cabin operator context and temporal hysteresis.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
from safevision.config import RiskEngineConfig
from safevision.context.danger_zone import DangerZone, DangerZoneManager, point_in_polygon, point_to_polygon_distance
from safevision.context.operator_context import OperatorContextEstimator, OperatorState
from safevision.context.ppe_association import WorkerPPEStatus
from safevision.tracking.byte_track import TrackedObject


class RiskLevel(Enum):
    SAFE = "SAFE"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class WorkerRiskAssessment:
    track_id: int
    raw_risk_score: float              # Unfiltered R_i(t) in [0.0, 1.0]
    smoothed_risk_score: float         # Post-hysteresis risk score
    risk_level: RiskLevel              # SAFE, WARNING, CRITICAL
    
    # Component breakdown: R_i(t) = G_i(t) * [w_z*Z_i + w_m*M_i + w_t*T_i + w_p*P_i]
    occupancy_Z: float                 # Z_i(t)
    motion_M: float                    # M_i(t)
    exposure_T: float                  # T_i(t)
    ppe_penalty_P: float               # P_i(t)
    operator_gate_G: float             # G_i(t)
    
    # Contextual metadata
    operator_state: OperatorState
    associated_machine_id: Optional[int]
    distance_to_machine: float         # in pixels
    exposure_frames: int               # consecutive frames in danger zone
    active_reasons: List[str]          # Human-readable explanations for alerts


class RiskEngine:
    """
    Evaluates temporal risk for each tracked worker in each frame.
    Enforces the mathematical formulation and hysteresis transitions from Slide 24.
    """
    def __init__(self, config: Optional[RiskEngineConfig] = None):
        self.config = config or RiskEngineConfig()
        
        # State tracking per worker
        self.exposure_duration: Dict[int, float] = {}       # track_id -> T_i in [0.0, 1.0]
        self.exposure_frame_count: Dict[int, int] = {}     # track_id -> integer count
        self.smoothed_scores: Dict[int, float] = {}
        self.previous_levels: Dict[int, RiskLevel] = {}
        
        # Hysteresis counters
        self.raise_counter: Dict[int, int] = {}
        self.clear_counter: Dict[int, int] = {}
        
        # Previous machine distances for relative approach velocity
        self.prev_distances: Dict[Tuple[int, int], float] = {}

        # FIX 3: grace-period bookkeeping
        self._last_seen: Dict[int, int] = {}
        self._tick = 0

    def reset(self):
        self._last_seen.clear()
        self._tick = 0
        self.exposure_duration.clear()
        self.exposure_frame_count.clear()
        self.smoothed_scores.clear()
        self.previous_levels.clear()
        self.raise_counter.clear()
        self.clear_counter.clear()
        self.prev_distances.clear()

    def evaluate(self,
                 workers: List[TrackedObject],
                 machinery: List[TrackedObject],
                 danger_zones: List[DangerZone],
                 ppe_statuses: Dict[int, WorkerPPEStatus],
                 operator_states: Dict[int, Tuple[OperatorState, float, Optional[int]]]) -> Dict[int, WorkerRiskAssessment]:
        """
        Computes risk scores and alerts for all active workers in current frame.
        """
        # FIX 3: keep a worker's temporal state through short dropouts; purge only after the grace period
        self._tick += 1
        for w in workers:
            self._last_seen[w.track_id] = self._tick
        stale = [tid for tid, seen in self._last_seen.items()
                 if self._tick - seen > self.config.state_grace_frames]
        for tid in stale:
            for d in (self.exposure_duration, self.exposure_frame_count, self.smoothed_scores,
                      self.previous_levels, self.raise_counter, self.clear_counter, self._last_seen):
                d.pop(tid, None)
            for key in [k for k in self.prev_distances if k[0] == tid]:
                del self.prev_distances[key]

        results: Dict[int, WorkerRiskAssessment] = {}

        for w in workers:
            tid = w.track_id
            w_center = w.center
            w_bottom = (w_center[0], float(w.bbox[3]))

            # 1. Danger Zone Occupancy Z_i(t)
            # FIX 5: score the worker against EVERY zone and keep the highest. Ties (e.g. all zero)
            # fall back to the nearest machine so M_i and distance still have a machine to refer to.
            Z_i = 0.0
            closest_machine_id = None
            min_dist_to_machine = float('inf')

            for dz in danger_zones:
                dist_c = float(np.hypot(w_bottom[0] - dz.centroid[0], w_bottom[1] - dz.centroid[1]))
                d_edge = point_to_polygon_distance(w_bottom[0], w_bottom[1], dz.polygon)
                if point_in_polygon(w_bottom[0], w_bottom[1], dz.polygon):
                    z = min(1.0, 0.65 + 0.35 * (d_edge / 50.0))      # penetration: 0.65 .. 1.0
                else:
                    buffer_px = 120.0
                    z = 0.50 * (1.0 - d_edge / buffer_px) if d_edge < buffer_px else 0.0
                if z > Z_i or (z == Z_i and dist_c < min_dist_to_machine):
                    Z_i = z
                    closest_machine_id = dz.machine_id
                    min_dist_to_machine = dist_c

            # 2. Observed Machinery Motion / Approach M_i(t)
            M_i = 0.0
            if closest_machine_id is not None:
                target_machine = next((m for m in machinery if m.track_id == closest_machine_id), None)
                if target_machine is not None:
                    # Machine velocity towards worker
                    m_speed = target_machine.speed
                    if m_speed > 0.5:
                        m_cx, m_cy = target_machine.center
                        vec_to_worker = np.array([w_bottom[0] - m_cx, w_bottom[1] - m_cy])
                        norm_vec = vec_to_worker / max(1e-4, np.linalg.norm(vec_to_worker))
                        vel_dir = target_machine.velocity / max(1e-4, m_speed)
                        
                        # Cosine similarity: +1 if machine moving towards worker
                        approach_cos = float(np.dot(vel_dir, norm_vec))
                        if approach_cos > 0.0:
                            # Scaling motion threat by speed and approach angle
                            M_i = min(1.0, (m_speed / 8.0) * approach_cos)

                    # Also compare distance change from previous frame
                    pair_key = (tid, closest_machine_id)
                    if pair_key in self.prev_distances:
                        delta_dist = self.prev_distances[pair_key] - min_dist_to_machine
                        if delta_dist > 1.0: # closing distance
                            approach_boost = min(0.3, delta_dist / 15.0)
                            M_i = min(1.0, M_i + approach_boost)
                    self.prev_distances[pair_key] = min_dist_to_machine

            # 3. Exposure Duration T_i(t)
            current_T = self.exposure_duration.get(tid, 0.0)
            current_fcount = self.exposure_frame_count.get(tid, 0)
            
            if Z_i >= 0.40:
                current_T = min(1.0, current_T + self.config.exposure_accumulation_rate)
                current_fcount += 1
            else:
                current_T = max(0.0, current_T - self.config.exposure_decay_rate)
                current_fcount = max(0, current_fcount - 1)

            self.exposure_duration[tid] = current_T
            self.exposure_frame_count[tid] = current_fcount
            T_i = current_T

            # 4. PPE Non-Compliance Penalty P_i(t)
            ppe_stat = ppe_statuses.get(tid, None)
            P_i = ppe_stat.ppe_penalty if ppe_stat else 0.5

            # 5. Operator Context Gate G_i(t)
            op_state, G_i, host_m_id = operator_states.get(
                tid, (OperatorState.PEDESTRIAN, 1.0, None)
            )

            # 6. Combined Raw Risk Calculation: R_i(t)
            # FIX 1: the operator gate silences machine-proximity terms only; PPE is always checked.
            proximity = (self.config.weight_danger_zone * Z_i +
                         self.config.weight_machine_motion * M_i +
                         self.config.weight_exposure_time * T_i)
            ppe_term = self.config.weight_ppe_noncompliance * P_i
            if self.config.gate_applies_to_ppe:
                weighted = G_i * (proximity + ppe_term)
            else:
                weighted = G_i * proximity + ppe_term
            # FIX 2: PPE evidence can raise an alert by itself once enough frames were observed.
            ppe_ready = (ppe_stat is not None and
                         len(ppe_stat.helmet_history) >= self.config.ppe_min_evidence_frames)
            ppe_floor = self.config.ppe_alert_scale * P_i if ppe_ready else 0.0
            raw_R = float(np.clip(max(weighted, ppe_floor), 0.0, 1.0))

            # 7. Temporal Smoothing & Hysteresis
            prev_level = self.previous_levels.get(tid, RiskLevel.SAFE)
            prev_smoothed = self.smoothed_scores.get(tid, raw_R)
            
            # Exponential moving average filter
            smoothed_R = 0.65 * raw_R + 0.35 * prev_smoothed

            # Determine proposed candidate level from smoothed score
            if smoothed_R >= self.config.threshold_critical:
                candidate_level = RiskLevel.CRITICAL
            elif smoothed_R >= self.config.threshold_warning:
                candidate_level = RiskLevel.WARNING
            else:
                candidate_level = RiskLevel.SAFE

            # Apply Hysteresis
            final_level = prev_level
            
            if candidate_level.value != prev_level.value:
                # Is this an escalation or de-escalation?
                is_escalation = (
                    (candidate_level == RiskLevel.CRITICAL and prev_level in (RiskLevel.WARNING, RiskLevel.SAFE)) or
                    (candidate_level == RiskLevel.WARNING and prev_level == RiskLevel.SAFE)
                )

                if is_escalation:
                    self.raise_counter[tid] = self.raise_counter.get(tid, 0) + 1
                    self.clear_counter[tid] = 0
                    if self.raise_counter[tid] >= self.config.hysteresis_raise_frames:
                        final_level = candidate_level
                        self.raise_counter[tid] = 0
                else:
                    self.clear_counter[tid] = self.clear_counter.get(tid, 0) + 1
                    self.raise_counter[tid] = 0
                    if self.clear_counter[tid] >= self.config.hysteresis_clear_frames:
                        final_level = candidate_level
                        self.clear_counter[tid] = 0
            else:
                self.raise_counter[tid] = 0
                self.clear_counter[tid] = 0

            self.previous_levels[tid] = final_level
            self.smoothed_scores[tid] = smoothed_R

            # 8. Human-Readable Explanation / Active Reasons
            reasons = []
            if op_state == OperatorState.CONFIRMED_OPERATOR:
                reasons.append("Confirmed Cabin Operator (Proximity Alerts Suppressed)")
            elif op_state == OperatorState.UNCERTAIN:
                reasons.append("Ambiguous Cabin State (Conservative Watch)")

            if Z_i >= 0.5:
                reasons.append("Occupying Machinery Dynamic Danger Zone")
            elif Z_i >= 0.25:
                reasons.append("Approaching Danger Zone Perimeter")

            if M_i >= 0.4:
                reasons.append("Machine Approaching / Heading Towards Worker")

            if T_i >= 0.35:
                reasons.append(f"Persistent Exposure ({current_fcount} frames in zone)")

            if ppe_stat:
                if not ppe_stat.has_helmet:
                    reasons.append("Missing Hardhat / Head Unprotected")
                if not ppe_stat.has_vest:
                    reasons.append("Missing High-Visibility Safety Vest")

            if not reasons:
                reasons.append("Safe Working Distance & Full PPE Verified")

            results[tid] = WorkerRiskAssessment(
                track_id=tid,
                raw_risk_score=raw_R,
                smoothed_risk_score=smoothed_R,
                risk_level=final_level,
                occupancy_Z=Z_i,
                motion_M=M_i,
                exposure_T=T_i,
                ppe_penalty_P=P_i,
                operator_gate_G=G_i,
                operator_state=op_state,
                associated_machine_id=closest_machine_id,
                distance_to_machine=min_dist_to_machine if min_dist_to_machine != float('inf') else 0.0,
                exposure_frames=current_fcount,
                active_reasons=reasons
            )

        return results
