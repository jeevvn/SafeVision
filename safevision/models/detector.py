"""
Dual-Detector Core and Modular Interface
Provides interfaces for Model A (Scene Detector: Workers & Machinery)
and Model B (PPE Detector: Helmets, Vests, Heads).
Supports real PyTorch/YOLO models as well as benchmark replay.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
from safevision.config import DetectorConfig


class BaseDetector:
    """Abstract base detector interface."""
    def detect(self, image: Union[Image.Image, np.ndarray]) -> List[Dict]:
        raise NotImplementedError


class SceneDetector(BaseDetector):
    """
    Model A: Scene Detector
    Responsible for identifying construction workers and machinery classes
    (e.g., excavator, dump truck, loader, crane, bulldozer).
    """
    def __init__(self, weights_path: Optional[str] = None, config: Optional[DetectorConfig] = None):
        import os
        if weights_path is None:
            default_w = os.path.join("safevision", "models", "weights", "scene_detector_best.pt")
            if os.path.exists(default_w):
                weights_path = default_w
            elif os.path.exists("yolov8n.pt"):
                weights_path = "yolov8n.pt"

        self.weights_path = weights_path
        self.config = config or DetectorConfig()
        self.model = None
        if weights_path and os.path.exists(weights_path):
            self._load_weights(weights_path)

    def _load_weights(self, weights_path: str):
        try:
            from ultralytics import YOLO
            self.model = YOLO(weights_path)
            print(f"[SceneDetector] Successfully loaded YOLO weights: {weights_path}")
        except Exception as e:
            print(f"[SceneDetector] Could not load YOLO model ({e}). Operating in modular inference mode.")

    def detect(self, image: Union[Image.Image, np.ndarray]) -> List[Dict]:
        if self.model is not None:
            results = self.model(image, verbose=False)
            dets = []
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names.get(cls_id, "unknown")
                    conf = float(box.conf[0])
                    if conf >= self.config.scene_conf_threshold:
                        xyxy = box.xyxy[0].cpu().numpy().tolist()
                        dets.append({
                            "bbox": xyxy,
                            "class_name": cls_name,
                            "score": conf,
                            "detector": "scene"
                        })
            return dets
        return []


class PPEDetector(BaseDetector):
    """
    Model B: PPE Detector
    Specialized for fine-grained PPE objects: helmet, vest, uncovered head.
    """
    def __init__(self, weights_path: Optional[str] = None, config: Optional[DetectorConfig] = None):
        import os
        if weights_path is None:
            default_w = os.path.join("safevision", "models", "weights", "ppe_detector_best.pt")
            if os.path.exists(default_w):
                weights_path = default_w

        self.weights_path = weights_path
        self.config = config or DetectorConfig()
        self.model = None
        if weights_path and os.path.exists(weights_path):
            self._load_weights(weights_path)

    def _load_weights(self, weights_path: str):
        try:
            from ultralytics import YOLO
            self.model = YOLO(weights_path)
            print(f"[PPEDetector] Successfully loaded YOLO weights: {weights_path}")
        except Exception as e:
            print(f"[PPEDetector] Could not load YOLO model ({e}). Operating in modular inference mode.")

    def detect(self, image: Union[Image.Image, np.ndarray]) -> List[Dict]:
        if self.model is not None:
            results = self.model(image, verbose=False)
            dets = []
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names.get(cls_id, "unknown")
                    conf = float(box.conf[0])
                    if conf >= self.config.ppe_conf_threshold:
                        xyxy = box.xyxy[0].cpu().numpy().tolist()
                        dets.append({
                            "bbox": xyxy,
                            "class_name": cls_name,
                            "score": conf,
                            "detector": "ppe"
                        })
            return dets
        return []


class DualDetectorPipeline:
    """
    Integrates Model A (Scene) and Model B (PPE) into a unified inference flow.
    """
    def __init__(self, scene_detector: Optional[SceneDetector] = None,
                 ppe_detector: Optional[PPEDetector] = None,
                 config: Optional[DetectorConfig] = None):
        self.scene_detector = scene_detector or SceneDetector(config=config)
        self.ppe_detector = ppe_detector or PPEDetector(config=config)
        self.config = config or DetectorConfig()

    def process_frame(self, image: Union[Image.Image, np.ndarray]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Executes both detectors and separates outputs:
        Returns:
            (worker_detections, machinery_detections, ppe_detections)
        """
        scene_dets = self.scene_detector.detect(image)
        ppe_dets = self.ppe_detector.detect(image)

        workers = []
        machinery = []
        for d in scene_dets:
            cname = d['class_name'].lower()
            if cname in ('worker', 'person', 'pedestrian'):
                workers.append(d)
            elif cname in ('excavator', 'dump_truck', 'wheel_loader', 'crane', 'bulldozer', 'truck', 'machinery', 'car', 'bus', 'vehicle'):
                machinery.append(d)

        # Also incorporate person detections from Model B (PPE detector) with IoU deduplication
        pure_ppe_dets = []
        for d in ppe_dets:
            cname = d['class_name'].lower()
            if cname in ('worker', 'person'):
                box_d = d['bbox']
                is_dup = False
                for w in workers:
                    box_w = w['bbox']
                    ix1 = max(box_d[0], box_w[0])
                    iy1 = max(box_d[1], box_w[1])
                    ix2 = min(box_d[2], box_w[2])
                    iy2 = min(box_d[3], box_w[3])
                    iw = max(0.0, ix2 - ix1)
                    ih = max(0.0, iy2 - iy1)
                    inter = iw * ih
                    if inter > 0:
                        union = (box_d[2]-box_d[0])*(box_d[3]-box_d[1]) + (box_w[2]-box_w[0])*(box_w[3]-box_w[1]) - inter
                        if union > 0 and (inter / union) >= 0.45:
                            is_dup = True
                            break
                if not is_dup:
                    workers.append(d)
            else:
                pure_ppe_dets.append(d)

        return workers, machinery, pure_ppe_dets
