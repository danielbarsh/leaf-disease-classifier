"""Robust PyTorch pipeline for leaf disease classification.

Standalone module addressing background bias, OOD overconfidence, healthy-
class imbalance, and unreliable predictions on ambiguous inputs. Independent
of the project's existing TensorFlow/Keras pipeline in src/ and app/.
"""

from .augmentation import get_train_transforms, get_val_transforms
from .calibration import ModelWithTemperature
from .inference import infer_with_rejection
from .losses import MultiClassFocalLoss
from .model import LeafDiseaseClassifier

__all__ = [
    "get_train_transforms",
    "get_val_transforms",
    "MultiClassFocalLoss",
    "LeafDiseaseClassifier",
    "ModelWithTemperature",
    "infer_with_rejection",
]
