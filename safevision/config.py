"""
SafeVision System Configuration
Defines hyperparameters, detection classes, risk weights, and thresholds
based on the DL Final Presentation specifications.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class RiskEngineConfig:
    # Formula weights: R_i(t) = G_i(t) * [w_z*Z_i(t) + w_m*M_i(t) + w_t*T_i(t) + w_p*P_i(t)]
    weight_danger_zone: float = 0.35      # w_z: Danger zone penetration depth / proximity
    weight_machine_motion: float = 0.25   # w_m: Observed machinery motion towards worker
    weight_exposure_time: float = 0.20    # w_t: Temporal accumulation of exposure duration
    weight_ppe_noncompliance: float = 0.20 # w_p: Lack of required PPE (helmet/vest)

    # Risk thresholds
    threshold_warning: float = 0.35       # Level 1 Alert (WARNING)
    threshold_critical: float = 0.65      # Level 2 Alert (CRITICAL)

    # Temporal dynamics
    exposure_accumulation_rate: float = 0.05  # T_i increase per frame when in danger zone
    exposure_decay_rate: float = 0.08         # T_i decrease per frame when in safe zone
    
    # Hysteresis parameters (persistence before raising/clearing alert)
    hysteresis_raise_frames: int = 3   # Consecutive frames required to escalate
    hysteresis_clear_frames: int = 5   # Consecutive frames below threshold to de-escalate

    # FIX 3: keep a worker's temporal state for this many frames after they disappear
    state_grace_frames: int = 15

    # FIX 2: PPE can raise an alert on its own. R >= ppe_alert_scale * P_i once enough evidence exists.
    # 0.6 * 0.6 (no helmet) = 0.36 -> WARNING.  Vest-only (0.4) = 0.24 -> no alert.  PPE alone never reaches CRITICAL.
    ppe_alert_scale: float = 0.60
    ppe_min_evidence_frames: int = 7   # frames of observation before PPE-only alerts are allowed

    # FIX 1: the operator gate only silences machine-proximity terms, never the PPE term
    gate_applies_to_ppe: bool = False


@dataclass
class DangerZoneConfig:
    # Base expansion margin around machine bounding box (proportional to machine size)
    base_margin_ratio: float = 0.40
    
    # Motion-adaptive expansion factor in the direction of machine movement
    velocity_expansion_factor: float = 2.5
    
    # Minimum speed threshold (pixels/frame) to trigger directional expansion
    # (FIX 4: velocity is now a smoothed 6-frame regression, so jitter no longer crosses this)
    min_speed_threshold: float = 1.5
    
    # Fixed circular radius for Baseline comparison (in pixels)
    baseline_fixed_radius: float = 120.0


@dataclass
class OperatorContextConfig:
    # Cabin region definition within machine bounding box (normalized [ymin, xmin, ymax, xmax])
    cabin_relative_box: Tuple[float, float, float, float] = (0.05, 0.20, 0.55, 0.80)
    
    # Minimum IoU or containment ratio to classify worker as inside cabin
    cabin_containment_threshold: float = 0.60
    
    # Maximum relative velocity between worker and machine to be considered seated operator
    max_relative_velocity: float = 2.0
    
    # Multiplier G_i(t) for confirmed cabin operator (suppresses external proximity alarm)
    operator_multiplier: float = 0.0
    
    # Multiplier G_i(t) for pedestrian or uncertain
    pedestrian_multiplier: float = 1.0

    # FIX 1: extra evidence required before someone is trusted to be the operator
    min_cabin_frames: int = 3                    # consecutive frames inside the cabin region
    min_inside_machine_ratio: float = 0.80       # worker box must sit inside the machine box
    max_worker_machine_height_ratio: float = 0.65  # a seated driver is much shorter than the machine


@dataclass
class PPEAssociationConfig:
    # Normalized vertical intervals relative to worker bounding box [0.0 = top, 1.0 = bottom]
    head_region_interval: Tuple[float, float] = (0.0, 0.35)
    torso_region_interval: Tuple[float, float] = (0.20, 0.75)
    
    # Minimum overlap / containment for positive association
    min_association_containment: float = 0.25
    
    # Temporal voting window length (in frames)
    voting_window_frames: int = 7
    voting_positive_threshold: float = 0.50

    # FIX 3: keep voting history this many frames after a worker disappears
    state_grace_frames: int = 15
    
    # Penalty values for missing items
    penalty_missing_helmet: float = 0.60
    penalty_missing_vest: float = 0.40


@dataclass
class DetectorConfig:
    # Model A: Scene detector classes (from MOCS)
    scene_classes: List[str] = field(default_factory=lambda: [
        "worker", "excavator", "dump_truck", "wheel_loader", "crane", "bulldozer", "roller"
    ])
    
    # Model B: PPE detector classes (from CHV / SHWD / SH17)
    ppe_classes: List[str] = field(default_factory=lambda: [
        "person", "helmet", "vest", "head_uncovered"
    ])
    
    # Confidence thresholds
    scene_conf_threshold: float = 0.40
    ppe_conf_threshold: float = 0.35
    iou_threshold: float = 0.45


@dataclass
class SystemConfig:
    risk: RiskEngineConfig = field(default_factory=RiskEngineConfig)
    danger_zone: DangerZoneConfig = field(default_factory=DangerZoneConfig)
    operator: OperatorContextConfig = field(default_factory=OperatorContextConfig)
    ppe: PPEAssociationConfig = field(default_factory=PPEAssociationConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    
    # Color palette (BGR for OpenCV, Hex for UI)
    colors_bgr: Dict[str, Tuple[int, int, int]] = field(default_factory=lambda: {
        "safe": (76, 175, 80),          # Green
        "warning": (33, 150, 243),      # Orange/Amber (in BGR: Blue=33, Green=150, Red=243)
        "critical": (36, 36, 230),      # Crimson Red
        "danger_zone": (0, 140, 255),   # Translucent Orange
        "danger_zone_active": (0, 0, 255), # Red
        "worker": (255, 191, 0),        # Deep Sky Blue
        "machinery": (235, 100, 52),    # Coral/Purple
        "helmet": (0, 255, 255),        # Yellow
        "vest": (0, 255, 0),            # Lime Green
        "head_uncovered": (0, 0, 255)   # Red
    })
