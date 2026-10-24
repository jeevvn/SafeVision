"""
Explainable Safety Event Logger and Audit Trail
Records time-stamped, structured safety events (warnings, critical alerts,
PPE infractions, cabin state changes) for real-time display and offline review.
"""

from dataclasses import asdict, dataclass
import json
from typing import Dict, List, Optional
import pandas as pd
from safevision.engine.risk_engine import RiskLevel, WorkerRiskAssessment


@dataclass
class SafetyEvent:
    frame_idx: int
    timestamp_sec: float
    worker_id: int
    machine_id: Optional[int]
    risk_level: str
    risk_score: float
    exposure_duration_sec: float
    causes: str
    operator_state: str
    z_occupancy: float
    m_motion: float
    t_exposure: float
    p_penalty: float


class EventLogger:
    """
    Maintains an audit log of safety events and provides structured summaries.
    """
    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.events: List[SafetyEvent] = []
        self._last_alert_level: Dict[int, RiskLevel] = {}

    def reset(self):
        self.events.clear()
        self._last_alert_level.clear()

    @property
    def event_records(self) -> List[SafetyEvent]:
        """Alias for backward compatibility."""
        return self.events

    def record_frame(self, frame_idx: int, assessments: Dict[int, WorkerRiskAssessment]):
        """
        Processes frame risk assessments and logs significant transitions
        (e.g., entered WARNING or CRITICAL, or persistent critical exposure).
        """
        timestamp_sec = frame_idx / max(1.0, self.fps)

        for tid, asm in assessments.items():
            curr_level = asm.risk_level
            prev_level = self._last_alert_level.get(tid, RiskLevel.SAFE)

            # Log when:
            # 1. State changed to WARNING or CRITICAL
            # 2. Or if CRITICAL persists (log periodically every 15 frames)
            should_log = False
            if curr_level != prev_level and curr_level != RiskLevel.SAFE:
                should_log = True
            elif curr_level == RiskLevel.CRITICAL and frame_idx % 15 == 0:
                should_log = True

            if should_log:
                exp_sec = asm.exposure_frames / max(1.0, self.fps)
                event = SafetyEvent(
                    frame_idx=frame_idx,
                    timestamp_sec=round(timestamp_sec, 2),
                    worker_id=tid,
                    machine_id=asm.associated_machine_id,
                    risk_level=curr_level.value,
                    risk_score=round(asm.smoothed_risk_score, 3),
                    exposure_duration_sec=round(exp_sec, 2),
                    causes="; ".join(asm.active_reasons),
                    operator_state=asm.operator_state.value,
                    z_occupancy=round(asm.occupancy_Z, 2),
                    m_motion=round(asm.motion_M, 2),
                    t_exposure=round(asm.exposure_T, 2),
                    p_penalty=round(asm.ppe_penalty_P, 2)
                )
                self.events.append(event)

            self._last_alert_level[tid] = curr_level

    def to_dataframe(self) -> pd.DataFrame:
        """Converts logged events to a pandas DataFrame for reporting."""
        if not self.events:
            return pd.DataFrame(columns=[
                "frame_idx", "timestamp_sec", "worker_id", "machine_id",
                "risk_level", "risk_score", "exposure_duration_sec",
                "causes", "operator_state", "z_occupancy", "m_motion", "t_exposure", "p_penalty"
            ])
        return pd.DataFrame([asdict(e) for e in self.events])

    def to_csv(self, filepath: str):
        """Exports event log to CSV file."""
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)

    def to_json(self, filepath: str):
        """Exports event log to JSON file."""
        with open(filepath, 'w') as f:
            json.dump([asdict(e) for e in self.events], f, indent=2)
