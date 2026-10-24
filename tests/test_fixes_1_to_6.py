"""
Comprehensive Unit Tests for Fixes 1 to 6 in SafeVision
"""

import sys
import os
from collections import deque
import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.abspath("."))

from safevision.config import (
    SystemConfig, RiskEngineConfig, OperatorContextConfig,
    DangerZoneConfig, PPEAssociationConfig
)
from safevision.context.danger_zone import DangerZone
from safevision.context.operator_context import OperatorContextEstimator, OperatorState
from safevision.context.ppe_association import PPEAssociator, WorkerPPEStatus
from safevision.engine.risk_engine import RiskEngine, RiskLevel
from safevision.tracking.byte_track import ByteTracker, KalmanBoxTracker, TrackedObject


def make_worker(track_id: int, bbox: np.ndarray, velocity: np.ndarray = None, class_name: str = "worker", score: float = 0.9):
    if velocity is None:
        velocity = np.array([0.0, 0.0])
    return TrackedObject(
        track_id=track_id,
        bbox=bbox,
        class_name=class_name,
        score=score,
        velocity=velocity,
        is_confirmed=True
    )


def test_fix_1_operator_context_strict_criteria_and_ppe_not_suppressed():
    """
    Fix 1:
    - Worker trusted as operator only if inside cabin, inside machine box, seated height ratio,
      min consecutive frames, and relative velocity check holds.
    - RiskEngine silences machine proximity terms for confirmed operator, but NOT PPE terms.
    """
    estimator = OperatorContextEstimator(grace_frames=15)
    # Machine box: [100, 100, 300, 300] (width 200, height 200)
    # Cabin relative box default: [0.15, 0.20, 0.55, 0.70] -> y: 130..210, x: 140..240
    machinery = [
        make_worker(
            track_id=1,
            bbox=np.array([100.0, 100.0, 300.0, 300.0]),
            class_name="excavator",
            score=0.95
        )
    ]

    # Case A: Standing person (height ratio = 150/200 = 0.75 > max 0.65)
    tall_worker = make_worker(
        track_id=10,
        bbox=np.array([150.0, 120.0, 200.0, 270.0]),
        class_name="worker",
        score=0.9
    )
    for _ in range(5):
        estimator.start_frame([10])
        state, mult, host_id = estimator.evaluate_operator_state(tall_worker, machinery)
    assert state != OperatorState.CONFIRMED_OPERATOR, "Standing worker should not be confirmed operator"

    # Case B: Moving relative to machine (rel_vel > 2.0)
    fast_worker = make_worker(
        track_id=11,
        bbox=np.array([150.0, 140.0, 200.0, 200.0]),
        class_name="worker",
        score=0.9,
        velocity=np.array([5.0, 0.0])
    )
    for _ in range(5):
        estimator.start_frame([11])
        state, mult, host_id = estimator.evaluate_operator_state(fast_worker, machinery)
    assert state != OperatorState.CONFIRMED_OPERATOR, "Fast moving worker relative to machine should not be operator"

    # Case C: Genuine seated operator inside cabin for 3+ frames
    operator_worker = make_worker(
        track_id=12,
        bbox=np.array([150.0, 140.0, 200.0, 200.0]), # height 60 / 200 = 0.30 <= 0.65
        class_name="worker",
        score=0.9,
        velocity=np.array([0.0, 0.0])
    )
    estimator.reset()
    for _ in range(2):
        estimator.start_frame([12])
        state, mult, _ = estimator.evaluate_operator_state(operator_worker, machinery)
    assert state != OperatorState.CONFIRMED_OPERATOR, "Should require min_cabin_frames (3)"

    estimator.start_frame([12])
    state, mult, host_id = estimator.evaluate_operator_state(operator_worker, machinery)
    assert state == OperatorState.CONFIRMED_OPERATOR, "Should be confirmed operator on 3rd frame"
    assert mult == 0.0, "Operator multiplier G_i should be 0.0"

    # Now verify in RiskEngine: operator gate G_i silences proximity, but PPE violation is NOT suppressed
    risk_cfg = RiskEngineConfig(gate_applies_to_ppe=False)
    engine = RiskEngine(config=risk_cfg)
    
    # Fake worker in danger zone with missing helmet
    ppe_stat = WorkerPPEStatus(
        track_id=12,
        has_helmet=False,
        has_vest=True,
        ppe_penalty=0.60,
        helmet_history=deque([False] * 10, maxlen=10),
        vest_history=deque([True] * 10, maxlen=10)
    )

    # Danger zone polygon around machine
    dz = DangerZone(
        machine_id=1,
        polygon=np.array([[50.0, 50.0], [350.0, 50.0], [350.0, 350.0], [50.0, 350.0]]),
        centroid=(200.0, 200.0),
        velocity=np.array([0.0, 0.0]),
        speed=0.0,
        is_expanding=False,
        expansion_angle_deg=0.0
    )

    op_states = {12: (OperatorState.CONFIRMED_OPERATOR, 0.0, 1)}
    ppe_statuses = {12: ppe_stat}

    # Evaluate multiple frames
    for _ in range(12):
        assessments = engine.evaluate(
            workers=[operator_worker],
            machinery=machinery,
            danger_zones=[dz],
            ppe_statuses=ppe_statuses,
            operator_states=op_states
        )
    asm = assessments[12]
    # G_i suppressed proximity (Z_i was > 0, but proximity term is gated by G_i=0)
    # However PPE floor or term (0.60 * 0.60 = 0.36) remains active!
    assert asm.risk_level == RiskLevel.WARNING, f"Expected WARNING due to PPE violation even though operator, got {asm.risk_level}"
    assert "Confirmed Cabin Operator (Proximity Alerts Suppressed)" in asm.active_reasons


def test_fix_2_ppe_alone_raises_warning_not_critical():
    """
    Fix 2:
    - Missing helmet alone raises WARNING once evidence >= 7 frames and hysteresis passes.
    - Missing vest alone (P_i = 0.4, 0.6 * 0.4 = 0.24) does not cross 0.35, so stays SAFE.
    - PPE alone never reaches CRITICAL (threshold 0.70).
    """
    engine = RiskEngine()
    worker = make_worker(
        track_id=1,
        bbox=np.array([500.0, 500.0, 550.0, 650.0]),
        class_name="worker",
        score=0.9
    )
    
    # 1. Helmet missing
    ppe_helmet_missing = WorkerPPEStatus(
        track_id=1,
        has_helmet=False,
        has_vest=True,
        ppe_penalty=0.60,
        helmet_history=deque(maxlen=20),
        vest_history=deque(maxlen=20)
    )

    # Under 7 evidence frames, raw_R stays at weighted sum (0.2 * 0.6 = 0.12)
    for f in range(6):
        ppe_helmet_missing.helmet_history.append(False)
        ppe_helmet_missing.vest_history.append(True)
        res = engine.evaluate([worker], [], [], {1: ppe_helmet_missing}, {1: (OperatorState.PEDESTRIAN, 1.0, None)})
        assert res[1].raw_risk_score < 0.30

    # At 7+ frames, ppe_floor = 0.60 * 0.60 = 0.36 kicks in
    for f in range(10):
        ppe_helmet_missing.helmet_history.append(False)
        ppe_helmet_missing.vest_history.append(True)
        res = engine.evaluate([worker], [], [], {1: ppe_helmet_missing}, {1: (OperatorState.PEDESTRIAN, 1.0, None)})

    assert res[1].risk_level == RiskLevel.WARNING
    assert res[1].smoothed_risk_score < 0.70, "PPE alone must not reach CRITICAL"

    # 2. Vest missing only: penalty is 0.40 -> ppe_floor = 0.6 * 0.4 = 0.24 < 0.35
    engine_vest = RiskEngine()
    ppe_vest_missing = WorkerPPEStatus(
        track_id=1,
        has_helmet=True,
        has_vest=False,
        ppe_penalty=0.40,
        helmet_history=deque([True] * 15, maxlen=20),
        vest_history=deque([False] * 15, maxlen=20)
    )
    for _ in range(15):
        res_v = engine_vest.evaluate([worker], [], [], {1: ppe_vest_missing}, {1: (OperatorState.PEDESTRIAN, 1.0, None)})
    assert res_v[1].risk_level == RiskLevel.SAFE, "Vest alone should remain SAFE (0.24 < 0.35)"


def test_fix_3_temporal_grace_period():
    """
    Fix 3:
    - When a worker disappears for <= 15 frames, PPE voting history and risk exposure states are retained.
    - After > 15 frames, the state is purged.
    """
    associator = PPEAssociator(config=PPEAssociationConfig(state_grace_frames=15))
    worker = make_worker(
        track_id=5,
        bbox=np.array([100.0, 100.0, 150.0, 200.0]),
        class_name="worker",
        score=0.9
    )
    helmet_det = {"bbox": [110.0, 95.0, 140.0, 120.0], "class_name": "helmet", "score": 0.9}
    
    # Frame 1: worker present with helmet
    associator.associate([worker], [helmet_det])
    assert 5 in associator.worker_ppe_states
    assert len(associator.worker_ppe_states[5].helmet_history) == 1

    # Absent for 10 frames (< 15)
    for _ in range(10):
        associator.associate([], [])
    assert 5 in associator.worker_ppe_states, "Worker state should be retained during grace frames"

    # Reappear at frame 12
    associator.associate([worker], [helmet_det])
    assert len(associator.worker_ppe_states[5].helmet_history) == 2, "History should be preserved"

    # Absent for 16 frames (> 15)
    for _ in range(16):
        associator.associate([], [])
    assert 5 not in associator.worker_ppe_states, "Worker state should be purged after grace period"


def test_fix_4_windowed_velocity_smoothing_and_deadband():
    """
    Fix 4:
    - Jitter below deadband (0.8 px/frame) returns 0 velocity.
    - Consistent movement over 6 frames yields smoothed velocity.
    """
    bbox = np.array([100.0, 100.0, 150.0, 200.0])
    tracker = KalmanBoxTracker(bbox=bbox, class_name="worker", score=0.9, track_id=1)

    # Apply minor jitter (< 0.5 pixels) over several frames
    for i in range(8):
        jitter = 0.3 * (1 if i % 2 == 0 else -1)
        tracker.predict()
        tracker.update(bbox + np.array([jitter, jitter, jitter, jitter]), score=0.9)
    
    assert np.all(tracker.velocity == 0.0), f"Jitter should be caught by deadband, got velocity {tracker.velocity}"

    # Now apply persistent motion (e.g. 5 pixels/frame in x)
    for i in range(10):
        shifted_bbox = bbox + np.array([5.0 * (i + 1), 0.0, 5.0 * (i + 1), 0.0])
        tracker.predict()
        tracker.update(shifted_bbox, score=0.9)
    
    assert tracker.velocity[0] > 2.0, f"Expected sustained x velocity, got {tracker.velocity[0]}"


def test_fix_5_multi_machine_danger_zone_evaluation():
    """
    Fix 5:
    - Worker is evaluated against all danger zones and receives the highest Z_i,
      even if another machine centroid is closer in Euclidean distance.
    """
    engine = RiskEngine()
    # Worker at (200, 200)
    worker = make_worker(
        track_id=1,
        bbox=np.array([190.0, 150.0, 210.0, 200.0]),
        class_name="worker",
        score=0.9
    )
    # Machine 1 is at (200, 100) -> distance to worker bottom (200, 200) is 100 px.
    # But its danger zone does NOT cover the worker (far polygon).
    dz1 = DangerZone(
        machine_id=1,
        polygon=np.array([[180.0, 0.0], [220.0, 0.0], [220.0, 50.0], [180.0, 50.0]]),
        centroid=(200.0, 100.0),
        velocity=np.array([0.0, 0.0]),
        speed=0.0,
        is_expanding=False,
        expansion_angle_deg=0.0
    )
    # Machine 2 is at (500, 500) -> centroid distance is ~424 px.
    # BUT Machine 2's danger zone directly encloses the worker at (200, 200)!
    dz2 = DangerZone(
        machine_id=2,
        polygon=np.array([[100.0, 100.0], [300.0, 100.0], [300.0, 300.0], [100.0, 300.0]]),
        centroid=(500.0, 500.0),
        velocity=np.array([0.0, 0.0]),
        speed=0.0,
        is_expanding=True,
        expansion_angle_deg=0.0
    )

    m1 = make_worker(track_id=1, bbox=np.array([180.0, 80.0, 220.0, 120.0]), class_name="roller", score=0.9)
    m2 = make_worker(track_id=2, bbox=np.array([450.0, 450.0, 550.0, 550.0]), class_name="loader", score=0.9)

    res = engine.evaluate([worker], [m1, m2], [dz1, dz2], {}, {1: (OperatorState.PEDESTRIAN, 1.0, None)})
    asm = res[1]
    assert asm.occupancy_Z >= 0.65, f"Worker should receive high penetration Z_i from dz2, got {asm.occupancy_Z}"
    assert asm.associated_machine_id == 2, f"Worker should be associated with machine 2 whose danger zone covers them, got {asm.associated_machine_id}"


def test_fix_6_tracker_id_independence():
    """
    Fix 6:
    - ByteTracker uses an instance-specific counter (_next_id) rather than a global/class variable.
    """
    tracker1 = ByteTracker(min_hits_to_confirm=1)
    tracker2 = ByteTracker(min_hits_to_confirm=1)

    dets = [{"bbox": [10.0, 10.0, 50.0, 50.0], "class_name": "worker", "score": 0.9}]
    
    t1_objs = tracker1.update(dets)
    t2_objs = tracker2.update(dets)

    assert len(t1_objs) == 1
    assert len(t2_objs) == 1
    assert t1_objs[0].track_id == 1
    assert t2_objs[0].track_id == 1, "Tracker 2 should start track_id from 1 independently"

    tracker1.reset()
    t1_reset_objs = tracker1.update(dets)
    assert t1_reset_objs[0].track_id == 1, "Tracker 1 should reset track_id back to 1"
