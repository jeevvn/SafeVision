"""
Generates real benchmark .mp4 video files from BenchmarkScenarioGenerator
so users can immediately test video auto-play, frame-by-frame tracking, and alerting.
"""
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from PIL import Image
from safevision.benchmark.scenario_generator import BenchmarkScenarioGenerator

def create_sample_videos():
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_videos")
    os.makedirs(out_dir, exist_ok=True)
    gen = BenchmarkScenarioGenerator(width=800, height=540)

    scenarios = [
        ("hazard_imminent_loader.mp4", gen.generate_scenario_3_imminent_danger(45), 15),
        ("hazard_ppe_violation.mp4", gen.generate_scenario_2_ppe_violation(35), 15),
        ("safe_operation.mp4", gen.generate_scenario_1_safe_operation(35), 15),
        ("cabin_operator_safe.mp4", gen.generate_scenario_4_cabin_operator(40), 15),
        ("transient_crossing.mp4", gen.generate_scenario_5_transient_crossing(35), 15),
    ]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    for fname, steps, fps in scenarios:
        fpath = os.path.join(out_dir, fname)
        w, h = steps[0].image.size
        writer = cv2.VideoWriter(fpath, fourcc, fps, (w, h))
        for step in steps:
            # Convert PIL RGB to OpenCV BGR
            arr_rgb = np.array(step.image)
            arr_bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
            writer.write(arr_bgr)
        writer.release()
        print(f"Generated {fpath} ({len(steps)} frames, {w}x{h} @ {fps}fps)")

if __name__ == "__main__":
    create_sample_videos()
