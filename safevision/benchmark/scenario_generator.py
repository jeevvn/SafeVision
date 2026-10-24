"""
Realistic Construction Safety Benchmark Scenario Generator
Synthesizes visual video frames and ground-truth interaction trajectories
for evaluation, ablation experiments, and interactive demonstrations.
"""

import os
from dataclasses import dataclass
from typing import Dict, Generator, List, Tuple
import numpy as np
from PIL import Image, ImageDraw


@dataclass
class ScenarioStep:
    frame_idx: int
    image: Image.Image
    worker_detections: List[Dict]
    machinery_detections: List[Dict]
    ppe_detections: List[Dict]
    ground_truth_event: str            # "SAFE", "WARNING_PPE", "CRITICAL_PROXIMITY", "OPERATOR_SAFE", "TRANSIENT_PASS"
    description: str


class BenchmarkScenarioGenerator:
    """
    Generates high-definition multi-frame construction site scenarios
    testing the edge cases defined in Slides 24-26.
    """
    def __init__(self, width: int = 800, height: int = 540):
        self.width = width
        self.height = height

    def _draw_construction_background(self) -> Image.Image:
        """Generates a realistic construction site backdrop."""
        img = Image.new("RGB", (self.width, self.height), (218, 198, 168)) # Sand/dirt ground
        draw = ImageDraw.Draw(img)
        
        # Sky
        draw.rectangle([0, 0, self.width, int(self.height * 0.35)], fill=(135, 175, 210))
        # Distant hills / horizon
        draw.polygon([(0, int(self.height * 0.35)),
                      (self.width * 0.3, int(self.height * 0.28)),
                      (self.width * 0.7, int(self.height * 0.32)),
                      (self.width, int(self.height * 0.30)),
                      (self.width, int(self.height * 0.35))], fill=(160, 150, 130))
        # Construction site gravel texture lines
        for y in range(int(self.height * 0.38), self.height, 40):
            draw.line([(0, y), (self.width, y + 10)], fill=(200, 180, 150), width=1)
        # Safety fence / barricade in background
        for x in range(20, self.width, 60):
            draw.line([(x, int(self.height * 0.34)), (x, int(self.height * 0.38))], fill=(255, 100, 0), width=3)
        draw.line([(0, int(self.height * 0.36)), (self.width, int(self.height * 0.36))], fill=(255, 140, 0), width=2)
        return img

    def generate_scenario_1_safe_operation(self, num_frames: int = 30) -> List[ScenarioStep]:
        """Scenario 1: Ground worker with full PPE working at safe distance from stationary excavator."""
        steps = []
        for f in range(num_frames):
            bg = self._draw_construction_background()
            draw = ImageDraw.Draw(bg)

            # Stationary Excavator at (120, 220) to (340, 420)
            m_box = [140.0, 220.0, 360.0, 420.0]
            draw.rectangle(m_box, fill=(190, 140, 40), outline=(100, 70, 20), width=3)
            # Cabin window
            draw.rectangle([210, 240, 290, 300], fill=(120, 160, 190), outline=(50, 50, 50), width=2)
            # Excavator boom/arm
            draw.line([(160, 260), (60, 180), (40, 340)], fill=(80, 80, 80), width=8)

            # Worker #1 at safe distance: x=580 + slight breathing movement, y=280 to 440
            w_x = 580.0 + np.sin(f * 0.2) * 2.0
            w_box = [w_x, 280.0, w_x + 55.0, 440.0]
            # Draw worker body
            draw.rectangle(w_box, fill=(60, 100, 160), outline=(30, 50, 90), width=2)
            # Hardhat on head
            h_box = [w_box[0] + 5.0, w_box[1] - 10.0, w_box[2] - 5.0, w_box[1] + 25.0]
            draw.rectangle(h_box, fill=(240, 200, 20), outline=(150, 120, 10), width=2)
            # Vest on torso
            v_box = [w_box[0] + 4.0, w_box[1] + 35.0, w_box[2] - 4.0, w_box[1] + 95.0]
            draw.rectangle(v_box, fill=(80, 220, 60), outline=(40, 140, 30), width=2)

            steps.append(ScenarioStep(
                frame_idx=f + 1,
                image=bg,
                worker_detections=[{"bbox": w_box, "class_name": "worker", "score": 0.92}],
                machinery_detections=[{"bbox": m_box, "class_name": "excavator", "score": 0.95}],
                ppe_detections=[
                    {"bbox": h_box, "class_name": "helmet", "score": 0.89},
                    {"bbox": v_box, "class_name": "vest", "score": 0.91}
                ],
                ground_truth_event="SAFE",
                description="Worker 1 in full PPE operating at a safe distance from stationary excavator."
            ))
        return steps

    def generate_scenario_2_ppe_violation(self, num_frames: int = 30) -> List[ScenarioStep]:
        """Scenario 2: Worker missing hardhat in safe zone -> WARNING alert due to PPE penalty."""
        steps = []
        for f in range(num_frames):
            bg = self._draw_construction_background()
            draw = ImageDraw.Draw(bg)

            # Stationary Roller
            m_box = [120.0, 240.0, 320.0, 430.0]
            draw.rectangle(m_box, fill=(160, 110, 40), outline=(80, 50, 20), width=3)
            draw.rectangle([180, 260, 260, 320], fill=(130, 170, 200), outline=(50, 50, 50), width=2)

            # Worker at safe distance, but NO HELMET (unprotected head)
            w_x = 550.0 + np.sin(f * 0.15) * 3.0
            w_box = [w_x, 280.0, w_x + 55.0, 440.0]
            draw.rectangle(w_box, fill=(70, 90, 140), outline=(30, 40, 70), width=2)
            # Bare head
            head_box = [w_box[0] + 8.0, w_box[1] - 5.0, w_box[2] - 8.0, w_box[1] + 25.0]
            draw.ellipse(head_box, fill=(210, 160, 130), outline=(140, 90, 70), width=1)
            # Vest present
            v_box = [w_box[0] + 4.0, w_box[1] + 35.0, w_box[2] - 4.0, w_box[1] + 95.0]
            draw.rectangle(v_box, fill=(80, 220, 60), outline=(40, 140, 30), width=2)

            steps.append(ScenarioStep(
                frame_idx=f + 1,
                image=bg,
                worker_detections=[{"bbox": w_box, "class_name": "worker", "score": 0.91}],
                machinery_detections=[{"bbox": m_box, "class_name": "roller", "score": 0.94}],
                ppe_detections=[
                    {"bbox": head_box, "class_name": "head_uncovered", "score": 0.86},
                    {"bbox": v_box, "class_name": "vest", "score": 0.88}
                ],
                ground_truth_event="WARNING_PPE",
                description="Worker in safe zone but missing hardhat (PPE infraction triggers warning)."
            ))
        return steps

    def generate_scenario_3_imminent_danger(self, num_frames: int = 40) -> List[ScenarioStep]:
        """
        Scenario 3: Wheel Loader moving towards an unprotected worker.
        Danger zone expands dynamically; risk smoothly escalates to CRITICAL.
        """
        steps = []
        for f in range(num_frames):
            bg = self._draw_construction_background()
            draw = ImageDraw.Draw(bg)

            # Moving Wheel Loader from left to right: x advances each frame
            m_start_x = 80.0 + f * 7.5
            m_box = [m_start_x, 210.0, m_start_x + 220.0, 420.0]
            draw.rectangle(m_box, fill=(210, 160, 30), outline=(120, 90, 20), width=3)
            # Bucket facing forward
            draw.polygon([(m_box[2], 340), (m_box[2] + 40, 340), (m_box[2] + 45, 420), (m_box[2], 420)],
                         fill=(80, 80, 80))
            # Wheels
            draw.ellipse([m_box[0] + 15, 380, m_box[0] + 75, 440], fill=(30, 30, 30))
            draw.ellipse([m_box[2] - 75, 380, m_box[2] - 15, 440], fill=(30, 30, 30))

            # Worker #3 standing at x=520, stationary, missing both helmet and vest!
            w_box = [520.0, 270.0, 575.0, 430.0]
            draw.rectangle(w_box, fill=(100, 60, 60), outline=(50, 30, 30), width=2)
            head_box = [w_box[0] + 8.0, w_box[1] - 5.0, w_box[2] - 8.0, w_box[1] + 25.0]
            draw.ellipse(head_box, fill=(210, 160, 130))

            gt_event = "SAFE" if f < 10 else ("WARNING" if f < 22 else "CRITICAL_PROXIMITY")

            steps.append(ScenarioStep(
                frame_idx=f + 1,
                image=bg,
                worker_detections=[{"bbox": w_box, "class_name": "worker", "score": 0.93}],
                machinery_detections=[{"bbox": m_box, "class_name": "wheel_loader", "score": 0.96}],
                ppe_detections=[{"bbox": head_box, "class_name": "head_uncovered", "score": 0.87}],
                ground_truth_event=gt_event,
                description=f"Moving Wheel Loader approaching unprotected worker (Frame {f+1}: distance decreasing)."
            ))
        return steps

    def generate_scenario_4_cabin_operator(self, num_frames: int = 35) -> List[ScenarioStep]:
        """
        Scenario 4: Worker is inside the excavator cabin driving the machine!
        Demonstrates the operator context G_i(t) novelty:
        - Baseline triggers FALSE ALARM because worker is within machine radius.
        - Proposed SafeVision detects operator containment -> G_i=0 -> Suppresses alert!
        """
        steps = []
        for f in range(num_frames):
            bg = self._draw_construction_background()
            draw = ImageDraw.Draw(bg)

            # Moving Excavator from left to right
            shift_x = f * 3.5
            m_box = [150.0 + shift_x, 200.0, 400.0 + shift_x, 420.0]
            draw.rectangle(m_box, fill=(190, 140, 30), outline=(100, 70, 20), width=3)
            
            # Cabin area
            cabin_box = [m_box[0] + 60.0, m_box[1] + 20.0, m_box[0] + 160.0, m_box[1] + 130.0]
            draw.rectangle(cabin_box, fill=(160, 200, 220), outline=(40, 40, 40), width=2)

            # Worker #4 is SEATED INSIDE THE CABIN
            w_box = [cabin_box[0] + 15.0, cabin_box[1] + 10.0, cabin_box[0] + 75.0, cabin_box[1] + 95.0]
            draw.rectangle(w_box, fill=(50, 80, 150), outline=(20, 40, 90), width=2)
            # Hardhat on operator
            h_box = [w_box[0] + 5.0, w_box[1] - 8.0, w_box[2] - 5.0, w_box[1] + 18.0]
            draw.rectangle(h_box, fill=(240, 200, 20))

            steps.append(ScenarioStep(
                frame_idx=f + 1,
                image=bg,
                worker_detections=[{"bbox": w_box, "class_name": "worker", "score": 0.89}],
                machinery_detections=[{"bbox": m_box, "class_name": "excavator", "score": 0.96}],
                ppe_detections=[{"bbox": h_box, "class_name": "helmet", "score": 0.85}],
                ground_truth_event="OPERATOR_SAFE",
                description="Cabin operator inside moving excavator. Proposed system suppresses alert (G_i=0), avoiding false alarm."
            ))
        return steps

    def generate_scenario_5_transient_crossing(self, num_frames: int = 30) -> List[ScenarioStep]:
        """
        Scenario 5: Worker briskly crosses past danger perimeter for only 2 frames.
        Baseline triggers a false jitter alarm.
        Proposed system uses temporal hysteresis to absorb transient crossing.
        """
        steps = []
        for f in range(num_frames):
            bg = self._draw_construction_background()
            draw = ImageDraw.Draw(bg)

            # Stationary Crane
            m_box = [180.0, 180.0, 360.0, 420.0]
            draw.rectangle(m_box, fill=(200, 120, 40), outline=(100, 60, 20), width=3)
            draw.line([(270, 180), (320, 60)], fill=(60, 60, 60), width=6)

            # Worker walks across from right to left, touching boundary around frames 14-15 only
            w_x = 650.0 - f * 16.0
            w_box = [w_x, 290.0, w_x + 55.0, 440.0]
            draw.rectangle(w_box, fill=(50, 110, 160), outline=(20, 60, 90), width=2)
            h_box = [w_box[0] + 5.0, w_box[1] - 8.0, w_box[2] - 5.0, w_box[1] + 20.0]
            draw.rectangle(h_box, fill=(240, 200, 20))
            v_box = [w_box[0] + 4.0, w_box[1] + 30.0, w_box[2] - 4.0, w_box[1] + 85.0]
            draw.rectangle(v_box, fill=(80, 220, 60))

            steps.append(ScenarioStep(
                frame_idx=f + 1,
                image=bg,
                worker_detections=[{"bbox": w_box, "class_name": "worker", "score": 0.91}],
                machinery_detections=[{"bbox": m_box, "class_name": "crane", "score": 0.95}],
                ppe_detections=[
                    {"bbox": h_box, "class_name": "helmet", "score": 0.88},
                    {"bbox": v_box, "class_name": "vest", "score": 0.90}
                ],
                ground_truth_event="TRANSIENT_PASS",
                description="Worker walking briskly past perimeter. Hysteresis prevents 1-frame false alarms."
            ))
        return steps

    def generate_real_dataset_sequence(self, dataset_name: str = "chv", num_frames: int = 20) -> List[ScenarioStep]:
        """
        Loads real construction site frames from local datasets (CHV or MOCS)
        and packages them as scenario steps for real-time YOLO dual-detector inference.
        """
        import os
        from safevision.data.dataset_loader import DatasetAuditor
        auditor = DatasetAuditor()
        steps = []

        if dataset_name == "mocs":
            target_dir = auditor.mocs_test_dir if os.path.exists(auditor.mocs_test_dir) else auditor.mocs_val_dir
            if not os.path.exists(target_dir):
                return self.generate_scenario_1_safe_operation(num_frames)
            files = sorted([f for f in os.listdir(target_dir) if f.lower().endswith((".jpg", ".png", ".jpeg")) and not f.startswith("._")])[:num_frames]
            for i, fname in enumerate(files):
                fpath = os.path.join(target_dir, fname)
                img = Image.open(fpath).convert("RGB")
                steps.append(ScenarioStep(
                    frame_idx=i + 1,
                    image=img,
                    worker_detections=None,
                    machinery_detections=None,
                    ppe_detections=None,
                    ground_truth_event="MOCS_REAL_CLIP",
                    description=f"Real MOCS Heavy Machinery Site Frame #{i+1} ({fname}). Live dual-detector YOLO inference."
                ))
        else:
            # CHV dataset
            if not os.path.exists(auditor.chv_images_dir):
                return self.generate_scenario_1_safe_operation(num_frames)
            for i in range(num_frames):
                sample = auditor.load_chv_sample(i)
                if sample:
                    img, boxes = sample
                    fname = getattr(img, "info", {}).get("filename", f"ppe_{i:04d}.jpg")
                    steps.append(ScenarioStep(
                        frame_idx=i + 1,
                        image=img,
                        worker_detections=None,
                        machinery_detections=None,
                        ppe_detections=None,
                        ground_truth_event="CHV_REAL_CLIP",
                        description=f"Real CHV Construction Worker Frame #{i+1} ({fname}). Live dual-detector YOLO inference."
                    ))

        return steps if steps else self.generate_scenario_1_safe_operation(num_frames)
