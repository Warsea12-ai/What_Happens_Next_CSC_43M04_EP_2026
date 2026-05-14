class TemporalTransformer(nn.Module):
    """Self-attention sur les frames + CLS pooling."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        max_frames: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, max_frames + 1, dim) * 0.02)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,        # pre-norm = entraînement plus stable
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, T, D)
        B, T, _ = x.shape
        cls = self.cls_token.expand(B, -1, -1)        # (B, 1, D)
        x = torch.cat([cls, x], dim=1)                # (B, T+1, D)
        x = x + self.pos_embed[:, : T + 1]
        x = self.dropout(x)
        x = self.encoder(x)                           # (B, T+1, D)
        return self.norm(x[:, 0])                     # CLS -> (B, D)


class cnnn_CLS(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False,
        max_frames: int = 16,
    ) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.temporal = TemporalTransformer(
            dim=feature_dim, num_heads=4, num_layers=2, max_frames=max_frames
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = video_batch.shape

        frames = video_batch.reshape(B * T, C, H, W)
        frame_features = self.backbone(frames)
        frame_features = torch.flatten(frame_features, 1)
        sequence = frame_features.view(B, T, -1)      # (B, T, 512)

        pooled = self.temporal(sequence)              # (B, 512)
        return self.classifier(pooled)