import torch.nn as nn
import timm
import torch


class ViTClassifier(nn.Module):
    def __init__(self, backbone_name, num_classes, dropout_rate=0.4):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True)
        in_features = self.backbone.num_features
        self.backbone.reset_classifier(0)  # Remove default classifier head

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

    def get_features(self, x):
        with torch.no_grad():
            feats = self.backbone.forward_features(x)
            feats = self.backbone.forward_head(
                feats, pre_logits=True
            )  # get final embedding
            return feats
