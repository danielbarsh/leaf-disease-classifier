"""Transfer-learning classifier: pretrained backbone + Dropout/Linear head."""

import torch
import torch.nn as nn

try:
    import timm
    _HAS_TIMM = True
except ImportError:
    _HAS_TIMM = False
    from torchvision import models as tv_models


class LeafDiseaseClassifier(nn.Module):
    """Pretrained backbone (features only) + a fresh classification head.

    The backbone is loaded with its own head stripped (num_classes=0 in
    timm, or Identity in the torchvision fallback) so it only ever returns
    a flat feature vector. Only the head below is task-specific, and is what
    trains from scratch even when the backbone starts frozen.
    """

    def __init__(
        self,
        num_classes: int,
        backbone: str = "convnext_tiny",
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.backbone_name = backbone

        if _HAS_TIMM:
            self.backbone = timm.create_model(
                backbone, pretrained=pretrained, num_classes=0, global_pool="avg"
            )
            feature_dim = self.backbone.num_features
        else:
            self.backbone, feature_dim = self._build_torchvision_backbone(backbone, pretrained)

        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, num_classes),
        )

        if freeze_backbone:
            self.freeze_backbone()

    @staticmethod
    def _build_torchvision_backbone(backbone: str, pretrained: bool):
        """Fallback for environments without timm installed."""
        weights_arg = "DEFAULT" if pretrained else None
        if backbone == "efficientnet_b0":
            net = tv_models.efficientnet_b0(weights=weights_arg)
            feature_dim = net.classifier[1].in_features
            net.classifier = nn.Identity()
        elif backbone == "convnext_tiny":
            net = tv_models.convnext_tiny(weights=weights_arg)
            feature_dim = net.classifier[2].in_features
            net.classifier = nn.Sequential(nn.Flatten())
        else:
            raise ValueError(
                f"Unsupported backbone '{backbone}' without timm installed. "
                "Install timm for the full model zoo, or use 'efficientnet_b0'/'convnext_tiny'."
            )
        return net, feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Call before a fine-tuning stage, typically with a lower LR."""
        for param in self.backbone.parameters():
            param.requires_grad = True
