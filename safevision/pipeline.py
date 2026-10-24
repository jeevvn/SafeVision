"""
SafeVision End-to-End Processing Pipeline
Orchestrates video frame ingestion, dual-detector inference, ByteTrack tracking,
PPE association, motion-adaptive danger zones, operator context gating,
temporal risk evaluation, explainable event logging, and visual rendering.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from safevision.config import SystemConfig
from safevision.context.danger_zone import DangerZone, DangerZoneManager
from safevision.context.operator_context import OperatorContextEstimator, OperatorState
from safevision.context.ppe_association import PPEAssociator, WorkerPPEStatus
from safevision.engine.event_logger import EventLogger
from safevision.engine.risk_engine import RiskEngine, RiskLevel, WorkerRiskAssessment
from safevision.models.detector import DualDetectorPipeline
from safevision.models.fusion import DetectionFusion
from safevision.tracking.byte_track import ByteTracker, TrackedObject
from safevision.visualization.annotator import VisualAnnotator


@dataclass
class PipelineFrameOutput:
    frame_idx: int
    annotated_image: Image.Image
    workers: List[TrackedObject]
    machinery: List[TrackedObject]
    danger_zones: List[DangerZone]
    ppe_statuses: Dict[int, WorkerPPEStatus]
    assessments: Dict[int, WorkerRiskAssessment]
    operator_states: Dict[int, Tuple[OperatorState, float, Optional[int]]]


class SafeVisionPipeline:
    """
    Complete video-inference pipeline as specified in Slide 20 of DL Final Presentation.
    """
    def __init__(self, config: Optional[SystemConfig] = None, use_baseline: bool = False):
        self.config = config or SystemConfig()
        self.use_baseline = use_baseline  # When True: static circular zone, no hysteresis, no cabin suppression

        # Core subsystems
        self.detectors = DualDetectorPipeline(config=self.config.detector)
        self.fusion = DetectionFusion(iou_thresh=self.config.detector.iou_threshold)
        self.tracker = ByteTracker(
            high_thresh=self.config.detector.scene_conf_threshold,
            low_thresh=0.20,
            match_thresh=0.70
        )
        self.ppe_associator = PPEAssociator(config=self.config.ppe)
        self.danger_zone_mgr = DangerZoneManager(config=self.config.danger_zone)
        self.operator_estimator = OperatorContextEstimator(config=self.config.operator,
                                                           grace_frames=self.config.risk.state_grace_frames)
        self.risk_engine = RiskEngine(config=self.config.risk)
        self.logger = EventLogger(fps=30.0)
        self.annotator = VisualAnnotator(config=self.config)

        self.frame_idx = 0

    def reset(self):
        """Resets all tracking, temporal accumulation, and event loggers."""
        self.tracker.reset()
        self.ppe_associator.reset()
        self.risk_engine.reset()
        self.operator_estimator.reset()
        self.logger.reset()
        self.frame_idx = 0

    def process_frame(self,
                      frame_input: Union[Image.Image, np.ndarray],
                      custom_detections: Optional[Tuple[List[Dict], List[Dict], List[Dict]]] = None,
                      render_annotation: bool = True) -> PipelineFrameOutput:
        """
        Executes a single processing step.
        If custom_detections is provided (e.g., from scenario replay or ground truth),
        bypasses raw detector inference for reproducible benchmark evaluation.
        """
        self.frame_idx += 1

        # 1. Image preparation
        if isinstance(frame_input, np.ndarray):
            pil_img = Image.fromarray(frame_input)
        else:
            pil_img = frame_input

        # 2. Dual-Detector Inference
        if custom_detections is not None:
            raw_workers, raw_machinery, raw_ppe = custom_detections
        else:
            raw_workers, raw_machinery, raw_ppe = self.detectors.process_frame(pil_img)

        # 3. Detection Fusion & NMS
        workers_fused, machinery_fused, ppe_fused = self.fusion.fuse(raw_workers, raw_machinery, raw_ppe)

        # 4. Multi-Object Tracking (ByteTrack)
        scene_detections = workers_fused + machinery_fused
        tracked_objects = self.tracker.update(scene_detections)

        active_workers = [t for t in tracked_objects if t.class_name in ('worker', 'person', 'pedestrian')]
        active_machinery = [t for t in tracked_objects if t.class_name not in ('worker', 'person', 'pedestrian')]

        # 5. Operator Context Estimation G_i(t)
        operator_states = {}
        self.operator_estimator.start_frame([w.track_id for w in active_workers])
        for w in active_workers:
            if self.use_baseline:
                # Baseline has NO cabin operator awareness! (Always pedestrian)
                operator_states[w.track_id] = (OperatorState.PEDESTRIAN, 1.0, None)
            else:
                op_state, G_i, host_m_id = self.operator_estimator.evaluate_operator_state(w, active_machinery)
                operator_states[w.track_id] = (op_state, G_i, host_m_id)

        # 6. Danger Zone Generation
        danger_zones: List[DangerZone] = []
        for m in active_machinery:
            if self.use_baseline:
                # Baseline uses static fixed-radius circular zone
                dz = self.danger_zone_mgr.generate_baseline_fixed_zone(m)
            else:
                # Proposed uses motion-adaptive directional polygon
                dz = self.danger_zone_mgr.generate_adaptive_zone(m)
            danger_zones.append(dz)

        # 7. PPE-to-Worker Geometric Association & Temporal Voting
        ppe_statuses = self.ppe_associator.associate(active_workers, ppe_fused)

        # 8. Temporal Risk Scoring Engine R_i(t)
        if self.use_baseline:
            # Baseline: single-frame decision without temporal accumulation or hysteresis
            orig_raise = self.risk_engine.config.hysteresis_raise_frames
            orig_clear = self.risk_engine.config.hysteresis_clear_frames
            orig_accum = self.risk_engine.config.exposure_accumulation_rate
            self.risk_engine.config.hysteresis_raise_frames = 1
            self.risk_engine.config.hysteresis_clear_frames = 1
            self.risk_engine.config.exposure_accumulation_rate = 1.0  # Instantaneous!
            assessments = self.risk_engine.evaluate(
                active_workers, active_machinery, danger_zones, ppe_statuses, operator_states
            )
            self.risk_engine.config.hysteresis_raise_frames = orig_raise
            self.risk_engine.config.hysteresis_clear_frames = orig_clear
            self.risk_engine.config.exposure_accumulation_rate = orig_accum
        else:
            assessments = self.risk_engine.evaluate(
                active_workers, active_machinery, danger_zones, ppe_statuses, operator_states
            )

        # 9. Audit Event Logging
        self.logger.record_frame(self.frame_idx, assessments)

        # 10. Visual Overlay Rendering
        if render_annotation:
            annotated_img = self.annotator.annotate_frame(
                frame_img=pil_img,
                workers=active_workers,
                machinery=active_machinery,
                danger_zones=danger_zones,
                ppe_statuses=ppe_statuses,
                assessments=assessments,
                operator_states=operator_states,
                frame_idx=self.frame_idx,
                show_danger_zones=True,
                show_ppe_boxes=True,
                show_velocity_vectors=not self.use_baseline,
                show_hud=True
            )
        else:
            annotated_img = pil_img

        return PipelineFrameOutput(
            frame_idx=self.frame_idx,
            annotated_image=annotated_img,
            workers=active_workers,
            machinery=active_machinery,
            danger_zones=danger_zones,
            ppe_statuses=ppe_statuses,
            assessments=assessments,
            operator_states=operator_states
        )
