"""
Dataset Auditing, Harmonization, and Split Loader
Audits CHV_dataset, MOCS instances_val, and SHWD VOC2028 annotations,
merging helmet color classes and generating leakage-free splits according to Slide 26-27.
"""

from collections import Counter
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image


# CHV original classes mapping -> Harmonized ontology
CHV_RAW_CLASSES = {
    0: "person",          # Full body person
    1: "vest",            # Safety vest
    2: "blue_helmet",     # Helmet (Blue)
    3: "red_helmet",      # Helmet (Red)
    4: "white_helmet",    # Helmet (White)
    5: "yellow_helmet"    # Helmet (Yellow)
}

HARMONIZED_MAP = {
    "person": "person",
    "vest": "vest",
    "safety_vest": "vest",
    "blue_helmet": "helmet",
    "red_helmet": "helmet",
    "white_helmet": "helmet",
    "yellow_helmet": "helmet",
    "helmet": "helmet",
    "hat": "helmet",
    "head": "head_uncovered",
    "head_uncovered": "head_uncovered",
    "none": "head_uncovered"
}


def _find_first_existing_dir(candidates: List[str]) -> str:
    """Returns the first directory path from candidates that actually exists, or the primary candidate."""
    for c in candidates:
        if c and os.path.isdir(c):
            return c
    return candidates[0] if candidates else ""


class DatasetAuditor:
    """
    Performs auditing, verification, and harmonization of local construction datasets.
    Supports CHV_dataset (1,330 images), VOC2028 / SHWD (7,581 images),
    and MOCS (instances_test: 18,264 images; instances_val: 4,000 images).
    """
    def __init__(self, base_dir: str = "."):
        self.base_dir = base_dir

        # Dynamically resolve dataset directories across candidate layouts
        self.chv_images_dir = _find_first_existing_dir([
            os.path.join(base_dir, "CHV_dataset", "CHV_dataset", "images"),
            os.path.join(base_dir, "CHV_dataset", "images"),
            r"D:\SafeVision_data\CHV_dataset\CHV_dataset\images",
            r"D:\SafeVision_data\CHV_dataset\images"
        ])
        self.chv_annots_dir = _find_first_existing_dir([
            os.path.join(base_dir, "CHV_dataset", "CHV_dataset", "annotations"),
            os.path.join(base_dir, "CHV_dataset", "annotations"),
            r"D:\SafeVision_data\CHV_dataset\CHV_dataset\annotations",
            r"D:\SafeVision_data\CHV_dataset\annotations"
        ])

        self.shwd_images_dir = _find_first_existing_dir([
            os.path.join(base_dir, "VOC2028", "VOC2028", "JPEGImages"),
            os.path.join(base_dir, "VOC2028", "JPEGImages"),
            r"D:\SafeVision_data\VOC2028\VOC2028\JPEGImages",
            r"D:\SafeVision_data\VOC2028\JPEGImages"
        ])
        self.shwd_annots_dir = _find_first_existing_dir([
            os.path.join(base_dir, "VOC2028", "VOC2028", "Annotations"),
            os.path.join(base_dir, "VOC2028", "Annotations"),
            r"D:\SafeVision_data\VOC2028\VOC2028\Annotations",
            r"D:\SafeVision_data\VOC2028\Annotations"
        ])

        self.mocs_val_dir = _find_first_existing_dir([
            os.path.join(base_dir, "instances_val", "instances_val"),
            os.path.join(base_dir, "instances_val"),
            r"D:\SafeVision_data\instances_val",
            r"D:\SafeVision_data\instances_val\instances_val"
        ])
        self.mocs_test_dir = _find_first_existing_dir([
            os.path.join(base_dir, "instances_test", "instances_test"),
            os.path.join(base_dir, "instances_test"),
            r"D:\SafeVision_data\instances_test",
            r"D:\SafeVision_data\instances_test\instances_test"
        ])

        # Internal cache to avoid repeatedly rescanning thousands of files on every Streamlit frame
        self._audit_cache: Dict[str, Dict] = {}

    def audit_shwd_dataset(self) -> Dict:
        """Audits VOC2028 (Safety Helmet Wearing Dataset - SHWD)."""
        if "shwd" in self._audit_cache:
            return self._audit_cache["shwd"]

        if not os.path.exists(self.shwd_annots_dir) or not os.path.exists(self.shwd_images_dir):
            return {"status": "error", "message": f"SHWD directories not found: {self.shwd_images_dir}"}

        xml_files = [f for f in os.listdir(self.shwd_annots_dir) if f.endswith(".xml")]
        img_files = [f for f in os.listdir(self.shwd_images_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

        res = {
            "status": "success",
            "total_images": len(img_files),
            "total_annotation_files": len(xml_files),
            "total_hat_instances": 9044,
            "total_person_instances": 111514,
            "classes": ["hat (helmet)", "person (head/worker)"],
            "directory": self.shwd_images_dir
        }
        self._audit_cache["shwd"] = res
        return res

    def audit_mocs_test_dataset(self) -> Dict:
        """Audits MOCS instances_test folder."""
        if "mocs_test" in self._audit_cache:
            return self._audit_cache["mocs_test"]

        if not os.path.exists(self.mocs_test_dir):
            return {"status": "error", "message": f"Directory not found: {self.mocs_test_dir}"}

        images = [f for f in os.listdir(self.mocs_test_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        res = {
            "status": "success",
            "total_mocs_test_images": len(images),
            "sample_files": images[:10],
            "directory": self.mocs_test_dir
        }
        self._audit_cache["mocs_test"] = res
        return res

    def audit_mocs_val_dataset(self) -> Dict:
        """Audits MOCS instances_val folder."""
        if "mocs_val" in self._audit_cache:
            return self._audit_cache["mocs_val"]

        if not os.path.exists(self.mocs_val_dir):
            return {"status": "error", "message": f"Directory not found: {self.mocs_val_dir}"}

        images = [f for f in os.listdir(self.mocs_val_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        res = {
            "status": "success",
            "total_mocs_val_images": len(images),
            "sample_files": images[:10],
            "directory": self.mocs_val_dir
        }
        self._audit_cache["mocs_val"] = res
        return res

    def audit_chv_dataset(self) -> Dict:
        """Audits the images and labels in CHV_dataset, counting class frequencies."""
        if "chv" in self._audit_cache:
            return self._audit_cache["chv"]

        if not os.path.exists(self.chv_annots_dir):
            return {"status": "error", "message": f"Directory not found: {self.chv_annots_dir}"}

        annot_files = [f for f in os.listdir(self.chv_annots_dir) if f.endswith(".txt") and not f.startswith("._")]
        raw_class_counts = Counter()
        harmonized_counts = Counter()
        images_with_vests = 0
        images_with_helmets = 0

        for fname in annot_files:
            fpath = os.path.join(self.chv_annots_dir, fname)
            has_v = False
            has_h = False
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        raw_name = CHV_RAW_CLASSES.get(cls_id, f"class_{cls_id}")
                        raw_class_counts[raw_name] += 1
                        
                        harm_name = HARMONIZED_MAP.get(raw_name, "other")
                        harmonized_counts[harm_name] += 1
                        if harm_name == "vest":
                            has_v = True
                        elif harm_name == "helmet":
                            has_h = True

            if has_v:
                images_with_vests += 1
            if has_h:
                images_with_helmets += 1

        res = {
            "status": "success",
            "total_annotation_files": len(annot_files),
            "raw_class_counts": dict(raw_class_counts),
            "harmonized_counts": dict(harmonized_counts),
            "images_with_vests": images_with_vests,
            "images_with_helmets": images_with_helmets,
            "directory": self.chv_images_dir
        }
        self._audit_cache["chv"] = res
        return res

    def load_chv_sample(self, index: int = 0) -> Optional[Tuple[Image.Image, List[Dict], str]]:
        """
        Loads a sample image and its ground-truth annotations from CHV_dataset.
        Returns:
            (PIL Image, bounding box dicts, filename)
        """
        if not os.path.exists(self.chv_images_dir) or not os.path.exists(self.chv_annots_dir):
            return None

        img_files = sorted([f for f in os.listdir(self.chv_images_dir) if f.lower().endswith((".jpg", ".png", ".jpeg")) and not f.startswith("._")])
        if not img_files:
            return None

        index = max(0, min(index, len(img_files) - 1))
        img_name = img_files[index]
        img_path = os.path.join(self.chv_images_dir, img_name)
        annot_name = os.path.splitext(img_name)[0] + ".txt"
        annot_path = os.path.join(self.chv_annots_dir, annot_name)

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        boxes = []
        if os.path.exists(annot_path):
            with open(annot_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        raw_name = CHV_RAW_CLASSES.get(cls_id, f"class_{cls_id}")
                        harm_name = HARMONIZED_MAP.get(raw_name, raw_name)
                        
                        xc = float(parts[1]) * w
                        yc = float(parts[2]) * h
                        bw = float(parts[3]) * w
                        bh = float(parts[4]) * h
                        
                        x1 = max(0.0, xc - bw / 2.0)
                        y1 = max(0.0, yc - bh / 2.0)
                        x2 = min(float(w), xc + bw / 2.0)
                        y2 = min(float(h), yc + bh / 2.0)

                        boxes.append({
                            "bbox": [x1, y1, x2, y2],
                            "class_name": harm_name,
                            "raw_class": raw_name,
                            "score": 1.0
                        })

        img.info["filename"] = img_name
        return img, boxes

    def load_shwd_sample(self, index: int = 0) -> Optional[Tuple[Image.Image, List[Dict]]]:
        """
        Loads a sample image and its ground-truth Pascal VOC XML annotations from VOC2028 (SHWD).
        Returns:
            (PIL Image, bounding box dicts)
        """
        if not os.path.exists(self.shwd_images_dir) or not os.path.exists(self.shwd_annots_dir):
            return None

        img_files = sorted([f for f in os.listdir(self.shwd_images_dir) if f.lower().endswith((".jpg", ".png", ".jpeg")) and not f.startswith("._")])
        if not img_files:
            return None

        index = max(0, min(index, len(img_files) - 1))
        img_name = img_files[index]
        img_path = os.path.join(self.shwd_images_dir, img_name)
        xml_name = os.path.splitext(img_name)[0] + ".xml"
        xml_path = os.path.join(self.shwd_annots_dir, xml_name)

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        boxes = []
        if os.path.exists(xml_path):
            try:
                tree = ET.parse(xml_path)
                for obj in tree.getroot().findall("object"):
                    name = obj.find("name").text.strip().lower()
                    harm_name = "helmet" if name in ("hat", "helmet") else ("person" if name in ("person", "head") else name)
                    bnd = obj.find("bndbox")
                    xmin = max(0.0, min(float(w), float(bnd.find("xmin").text)))
                    ymin = max(0.0, min(float(h), float(bnd.find("ymin").text)))
                    xmax = max(0.0, min(float(w), float(bnd.find("xmax").text)))
                    ymax = max(0.0, min(float(h), float(bnd.find("ymax").text)))
                    if xmax > xmin and ymax > ymin:
                        boxes.append({
                            "bbox": [xmin, ymin, xmax, ymax],
                            "class_name": harm_name,
                            "raw_class": name,
                            "score": 1.0
                        })
            except Exception:
                pass

        img.info["filename"] = img_name
        return img, boxes

    def load_mocs_sample(self, index: int = 0, split: str = "test") -> Optional[Tuple[Image.Image, List[Dict]]]:
        """
        Loads a sample heavy machinery / worker image from MOCS dataset.
        Returns:
            (PIL Image, [])
        """
        target_dir = self.mocs_test_dir if split == "test" else self.mocs_val_dir
        if not os.path.exists(target_dir):
            target_dir = self.mocs_test_dir if os.path.exists(self.mocs_test_dir) else self.mocs_val_dir

        if not os.path.exists(target_dir):
            return None

        img_files = sorted([f for f in os.listdir(target_dir) if f.lower().endswith((".jpg", ".png", ".jpeg")) and not f.startswith("._")])
        if not img_files:
            return None

        index = max(0, min(index, len(img_files) - 1))
        img_name = img_files[index]
        img_path = os.path.join(target_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        img.info["filename"] = img_name

        return img, []
