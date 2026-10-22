"""Shared contracts, data schemas, and interfaces for Agri_Sorter."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

FRUITS = ["apple", "guava", "orange", "mango"]
EXTENDED_FRUITS = ["apple", "guava", "orange", "mango", "pomegranate"]
LEGACY_FRUITS = ["banana"]
CONDITIONS = ["fresh", "rotten", "bruised", "scarred", "bug_bite"]

COMPOUND_CLASSES = [
    f"{fruit}_{cond}" for fruit in FRUITS for cond in CONDITIONS
]
EXTENDED_COMPOUND_CLASSES = [
    f"{fruit}_{cond}" for fruit in EXTENDED_FRUITS for cond in CONDITIONS
]

STATUS_FRESH = "FRESH"
STATUS_REJECT = "REJECT"


def is_fresh_class(label: str) -> bool:
    """Check if class label denotes fresh quality."""
    return label.endswith("_fresh") or label == "fresh"


def is_defect_class(label: str) -> bool:
    """Check if class label denotes a defect."""
    return not is_fresh_class(label)


def parse_compound_label(label: str) -> Tuple[str, str]:
    """Split compound label into fruit type and defect condition."""
    if "_" in label:
        parts = label.split("_", 1)
        if parts[0] in FRUITS or parts[0] in EXTENDED_FRUITS or parts[0] in LEGACY_FRUITS:
            return parts[0], parts[1]
    # Fallback default
    return "unknown", label


@dataclass(frozen=True)
class FruitDetection:
    """Single frame object detection box and classification."""

    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2 (pixel coordinates)
    confidence: float
    class_id: int
    class_name: str
    fruit_type: str
    condition: str
    is_defect: bool


@dataclass
class TrackState:
    """State representation of a tracked fruit across conveyor frames."""

    track_id: int
    bbox: Tuple[int, int, int, int]
    fruit_type: str
    status: str = STATUS_FRESH
    defect_classes: Set[str] = field(default_factory=set)
    confidence: float = 1.0
    consecutive_defect_count: int = 0
    has_egressed: bool = False
    missed_frames: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    current_condition: str = "fresh"
    total_frames: int = 1

    @property
    def is_reject(self) -> bool:
        return self.status == STATUS_REJECT

    @property
    def is_fresh(self) -> bool:
        return self.status == STATUS_FRESH

    @property
    def centroid(self) -> Tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0)


@dataclass(frozen=True)
class FruitEvent:
    """Final fruit record pushed to FIFO queue upon leaving camera view."""

    track_id: int
    fruit_type: str
    final_status: str  # "FRESH" or "REJECT"
    defect_classes: List[str]  # All defect classes observed during transit
    confidence: float
    egress_timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "track_id": self.track_id,
            "fruit_type": self.fruit_type,
            "final_status": self.final_status,
            "defect_classes": ",".join(self.defect_classes) if self.defect_classes else "none",
            "confidence": round(self.confidence, 4),
            "egress_timestamp": self.egress_timestamp,
        }


def load_app_config(config_path: Optional[str or Path] = None) -> Dict[str, Any]:
    """Load configuration dictionary from YAML file."""
    if config_path is None:
        config_path = REPO_ROOT / "config" / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg
