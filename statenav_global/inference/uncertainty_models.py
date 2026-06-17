import math

import torch
import torch.nn as nn
import torch.nn.functional as F
# from mmcv.ops import MultiScaleDeformableAttention
import torchvision.models as models


# -------------------------
# Spatial Attention Module
# -------------------------
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, C, H, W)
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)    # (B,1,H,W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B,1,H,W)
        x_cat = torch.cat([avg_out, max_out,], dim=1)    # (B,2,H,W)
        attention = self.sigmoid(self.conv(x_cat))      # (B,1,H,W)
        return x * attention

# -------------------------
# Simple ResBlock
# -------------------------
class SimpleResBlock(nn.Module):
    """
    Lightweight depthwise-separable downsampling backbone (two stride-2 stages).
    Typical input: (B, 1, 17, 17) -> output: (B, 512, 5, 5).
    """

    def __init__(self, input_channels, output_channels=512, dropout_p=0.1, width=128):
        super().__init__()
        self.output_channels = output_channels
        self.block1 = self._make_stage(input_channels, width, stride=2, dropout_p=dropout_p)
        self.block2 = self._make_stage(width, output_channels, stride=2, dropout_p=dropout_p)

    @staticmethod
    def _make_stage(in_channels, out_channels, stride, dropout_p):
        """Depthwise 3x3 (stride) + pointwise 1x1 + residual shortcut."""
        stage = nn.Module()
        stage.dw = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride,
            padding=1, groups=in_channels, bias=False,
        )
        stage.bn_dw = nn.BatchNorm2d(in_channels)
        stage.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        stage.bn_pw = nn.BatchNorm2d(out_channels)
        stage.dropout = nn.Dropout2d(dropout_p)
        if in_channels != out_channels or stride != 1:
            stage.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            stage.shortcut = nn.Identity()
        return stage

    @staticmethod
    def _forward_stage(x, stage):
        residual = stage.shortcut(x)
        out = F.relu(stage.bn_dw(stage.dw(x)))
        out = stage.dropout(stage.bn_pw(stage.pw(out)))
        return F.relu(out + residual)

    def forward(self, x):
        x = self._forward_stage(x, self.block1)
        x = self._forward_stage(x, self.block2)
        return x

# -------------------------
# Fourier Scalar
# -------------------------
def _fourier_scalar(x, num_freqs):
    """Fourier features for one scalar (B, 1)."""
    freqs = (2.0 ** torch.arange(num_freqs, device=x.device, dtype=x.dtype)) * math.pi
    xc = x * freqs.unsqueeze(0)
    return torch.cat([x, torch.sin(xc), torch.cos(xc)], dim=-1)

# -------------------------
# MLP Fourier Fusion 
# -------------------------
class MLPFourierFusionDecoder(nn.Module):
    """MLP decoder: Fourier(cmd_v) + Fourier(cmd_w) fused with multi-stat visual pooling."""

    def __init__(self, embed_dim=512, num_freqs=16, dropout_p=0.1, predict_logstd=False):
        super().__init__()
        self.num_freqs = num_freqs
        self.embed_dim = embed_dim
        # Visual: embed_dim (spatial avg); command branch: embed_dim
        # Choose to use spatial avg or multi-stat visual pooling
        # if use multi-stat visual pooling, fusion_in = embed_dim * 4
        fusion_in = embed_dim * 2
        cmd_in = 2 * (num_freqs * 2 + 1)
        self.cmd_proj = nn.Sequential(
            nn.Linear(cmd_in, embed_dim), nn.Tanh(), nn.Dropout(p=dropout_p),
            nn.Linear(embed_dim, embed_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, embed_dim), nn.ReLU(), nn.Dropout(p=dropout_p),
            nn.Linear(embed_dim, 256), nn.BatchNorm1d(256), nn.Tanh(), nn.Dropout(p=dropout_p),
            nn.Linear(256, 64),
        )
        self.fc_pred = nn.Sequential(
            nn.Linear(64, 32), 
            nn.ReLU(),
            nn.Dropout(p=dropout_p), 
            nn.Linear(32, 16),
            nn.Dropout(p=dropout_p),
            nn.Linear(16, 1),
        )
        self.fc_logstd = (
            nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(32, 16),
            nn.Dropout(p=dropout_p),
            nn.Linear(16, 1),
            )
            if predict_logstd
            else None
        )

    def forward(self, features, commands):
        # features: (B, C, H, W); commands: (B, 2) => [v, omega]
        # x_cmd Shape [64,512]
        # feature_Shape [64, 512, 5, 5]]
        
        v, w = commands[:, :1], commands[:, 1:2]
        x_cmd = self.cmd_proj(torch.cat([_fourier_scalar(v, self.num_freqs), _fourier_scalar(w, self.num_freqs),], dim=1))
        
        x_vis = features.flatten(2).mean(dim=2)
        x = self.fusion(torch.cat([x_vis, x_cmd], dim=1))
        pred = self.fc_pred(x)
        return (pred, self.fc_logstd(x)) if self.fc_logstd is not None else pred

# -------------------------
# Elevation Only Network MSE
# -------------------------
class ElevationOnlyNetworkMSE(nn.Module):
    def __init__(
        self,
        embed_dim=512,
        num_heads=8,
        num_levels=1,
        num_points=4,
        dropout_p=0.1,
        freeze_backbone=False,
        num_fourier_freqs=16,
    ):
        super().__init__()

        # 1. Feature Extractor
        self.feature_extractor = SimpleResBlock(
            input_channels=1,
            output_channels=512,
            dropout_p=dropout_p)

        # 2. Spatial attention modules (optional usage)
        self.spatial_attention = SpatialAttention(kernel_size=3)
      
        # 3. Fusion Decoder
        self.mlp_fourier_decoder = MLPFourierFusionDecoder(embed_dim, num_fourier_freqs, dropout_p, predict_logstd=False,)

    def forward(self, elevation_map, commands):
        """
        Args:
            elevation_map: (B, 1, H, W)
            commands:      (B, 3)  # e.g. [v, omega, terrain_level]
        """
        # Feature extraction
        features = self.feature_extractor(elevation_map)  # => (B, E, H', W') or (B, 512, H', W')
        B, C, H_out, W_out = features.shape

        # spatial attention
        features = self.spatial_attention(features)
        
        # Fusion Decoder
        return self.mlp_fourier_decoder(features, commands)

# -------------------------
# Elevation Only Network MLL
# -------------------------
class ElevationOnlyNetworkMLL(nn.Module):
    def __init__(
        self,
        embed_dim=512,
        num_heads=8,
        num_levels=1,
        num_points=4,
        dropout_p=0.2,
        freeze_backbone=False,
        num_fourier_freqs=16,
    ):
        super().__init__()

        # 1. Feature Extractor
        self.feature_extractor = SimpleResBlock(
                input_channels=1,
                output_channels=512,
                dropout_p=dropout_p)

        # 2. Attention Modules (optional)
        self.spatial_attention = SpatialAttention(kernel_size=3)

        # 3. Fusion and Decoder
        self.mlp_fourier_decoder = MLPFourierFusionDecoder(embed_dim, num_fourier_freqs, dropout_p, predict_logstd=True)
      
    def forward(self, elevation_map, commands):
        """
        Returns: pred_mean, pred_logstd
        elevation_map: (B, 1, H, W)
        commands:      (B, 3)  # e.g. [v, omega, terrain_level]
        """
     
        # Feature extraction
        features = self.feature_extractor(elevation_map)
        B, C, H_out, W_out = features.shape

        # Optionally apply attention
        features = self.spatial_attention(features)

        # Fusion Decoder
        return self.mlp_fourier_decoder(features, commands)
       
            