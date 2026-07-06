"""ResNet18 implementation for CIFAR-10 classification."""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    pass


class PositionalEmbedding(nn.Module):
    pe: torch.Tensor

    def __init__(self, dim: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, dim)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe = self.pe[:, : x.size(1)].to(self.pe.device)
        return x + pe


class PatchEmbedding(nn.Module):
    """Patch embedding layer for Vision Transformer.

    Splits an image into fixed-size patches and projects each patch
    into an embedding space. For CIFAR-10 (32×32), patch_size=4 yields
    an 8×8 = 64 patch sequence.
    """

    def __init__(self, patch_size: int = 4, in_channels: int = 3, embed_dim: int = 64):
        super().__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.positional_embedding = PositionalEmbedding(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → proj → (B, embed_dim, H/patch_size, W/patch_size)
        out = self.proj(x)
        # Flatten spatial dims: (B, embed_dim, num_patches) → transpose → (B, num_patches, embed_dim)
        out = out.flatten(2).transpose(1, 2)
        out = self.positional_embedding(out)
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(8, out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, out_c),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.out_linear = nn.Linear(d_model, d_model)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_size = x.size(0)

        # Q, K, V matrislerinin üçünü de aynı 'x' girdisinden üretiyoruz! 🎭
        # Boyut değişimi: [B, Seq_Len, d_model] -> [B, Seq_Len, num_heads, d_k] -> [B, num_heads, Seq_Len, d_k]
        Q = (
            self.q_linear(x)
            .view(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        K = (
            self.k_linear(x)
            .view(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        V = (
            self.v_linear(x)
            .view(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )

        # Kelimelerin birbiriyle olan çarpım skorları 📈
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Maskeleme (isteğe bağlı)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Dikkat ağırlıkları (Olasılık dağılımı)
        attention_weights = F.softmax(scores, dim=-1)

        # Ağırlıklarla V değerlerini çarpıyoruz 🤝
        context = torch.matmul(attention_weights, V)

        # Kafaları birleştirme (Concat) 🔗
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )

        # Son lineer katman
        output = self.out_linear(context)

        return output


class MLP(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * d_model, d_model)
        self.drop = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.multi_head_attn = MultiHeadAttention(d_model, n_heads)
        self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.mlp = MLP(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.multi_head_attn(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x


class MLPHead(nn.Module):
    def __init__(self, d_model: int, num_classes: int):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class VisionTransformer(nn.Module):
    """Vision Transformer (ViT) adapted for CIFAR-10 (32×32 images).

    CIFAR-10 specific configuration:
        - image_size=32, patch_size=4 → 8×8 = 64 patches per image
        - Default: d_model=768, n_heads=12, n_layers=12 (ViT-B like)

    To use a smaller variant for CIFAR-10, adjust via model_factory:
        - tiny:  d_model=192, n_heads=3, n_layers=3
        - small: d_model=384, n_heads=6, n_layers=6
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        num_classes: int = 10,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
    ):
        super().__init__()

        assert image_size % patch_size == 0, (
            "image_size must be divisible by patch_size"
        )
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_size = patch_size
        self.d_model = d_model

        self.patch_embed = PatchEmbedding(
            patch_size=patch_size, in_channels=3, embed_dim=d_model
        )

        # Pre-norm architecture: each block is [LayerNorm → attn/MLP → residual]
        self.transformer = nn.ModuleList(
            [
                TransformerBlock(d_model=d_model, n_heads=n_heads)
                for _ in range(n_layers)
            ]
        )

        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.mlp_head = MLPHead(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 32, 32)
        x = self.patch_embed(x)  # (B, num_patches, d_model)
        for block in self.transformer:
            x = block(x)
        x = self.norm(x)
        x = x.mean(dim=1)  # CLS-style pooling → (B, d_model)
        x = self.mlp_head(x)  # (B, num_classes)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int) -> None:
        super().__init__()

        self.block = ConvBlock(in_c, out_c, stride)

        self.skip = nn.Identity()
        if stride != 1 or in_c != out_c:
            self.skip = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.GroupNorm(8, out_c),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.block(x) + self.skip(x))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation (SE) block for channel-wise attention.

    This block adaptively recalibrates channel-wise feature responses by
    explicitly modeling interdependencies between channels. It consists of:
        - Squeeze: Global average pooling to create a channel descriptor.
        - Excitation: Two fully connected layers to capture channel relationships.
        - Scale: Reweight the original feature maps by the learned channel weights.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        """Initialize SEBlock.

        Args:
            channels: Number of input channels.
            reduction: Reduction ratio for the bottleneck in the excitation step.
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SE block."""
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResNet18(nn.Module):
    """ResNet-18 network architecture for CIFAR-10.

    Architecture overview:
        - Initial stem: 3x3 conv (64 ch)
        - Stage 1 (C1): 2x basic blocks, base channels 64, output 64 ch
        - Stage 2 (C2): 2x basic blocks, base channels 128, output 128 ch
        - Stage 3 (C3): 2x basic blocks, base channels 256, output 256 ch
        - Stage 4 (C4): 2x basic blocks, base channels 512, output 512 ch
        - Classification: global average pool + linear

    Total layers: 1 (stem conv) + 8 (basic blocks * 2 conv) + 1 (fc) = 18 conv layers
    """

    def __init__(
        self,
        num_classes: int = 10,
        block: type[nn.Module] = BasicBlock,
        layers: list[int] = [2, 2, 2, 2],
        zero_init_residual: bool = False,
    ) -> None:
        """Initialize ResNet18.

        Args:
            num_classes: Number of output classes (default 10 for CIFAR-10).
            block: Block type to use (default BasicBlock).
            layers: Number of basic blocks in each stage.
            zero_init_residual: If True, initialize last BN in each block to zero.
        """
        super().__init__()

        # Initial stem (CIFAR-friendly: no max pool to preserve small feature maps)
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Define channels for each stage
        self.layer1_channels = 64  # output = 64 * 1 = 64
        self.layer2_channels = 128  # output = 128 * 1 = 128
        self.layer3_channels = 256  # output = 256 * 1 = 256
        self.layer4_channels = 512  # output = 512 * 1 = 512

        # Stages: stride=2 for downsampling on first block of each stage
        self.layer1 = block(self.in_channels, self.layer1_channels, stride=1)
        self.layer2 = block(self.layer1_channels, self.layer2_channels, stride=2)
        self.layer3 = block(self.layer2_channels, self.layer3_channels, stride=2)
        self.layer4 = block(self.layer3_channels, self.layer4_channels, stride=2)

        self.stages = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        )

        # Final classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        # Weight initialization
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """Forward pass through ResNet18.

        Args:
            x: Input tensor (B, 3, H, W).
            return_features: If True, return feature maps before classification head.

        Returns:
            Classification logits (B, num_classes) or feature maps (B, C, H, W).
        """

        if x.dim() != 4:
            raise ValueError("Expected 4D input tensor (B, C, H, W)")

        # Initial stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Stages
        x = self.stages(x)

        if return_features:
            return x

        # Classification head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class SuperNet(nn.Module):
    def __init__(self, num_experts: int = 20, num_classes: int = 10):
        super().__init__()

        self.num_experts = num_experts

        # 🧠 Gating network
        self.gating_classifier = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.AdaptiveAvgPool2d((1, 1)),  # 🔥 FIX (çok önemli)
            nn.Flatten(),
            nn.Linear(128, 512),
            nn.SiLU(inplace=True),
            nn.Linear(512, num_experts),
            nn.Softmax(dim=1),
        )

        # 🧠 Experts (her biri feature extractor)
        self.experts = nn.ModuleList([ResNet18() for _ in range(num_experts)])

        # 🧠 final head
        self.head = nn.Sequential(
            nn.Linear(512, 512), nn.SiLU(inplace=True), nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)

        # 🎯 routing scores
        gate = self.gating_classifier(x)  # (B, E)

        # 🔥 hard routing: top-1 expert seç
        idx = gate.argmax(dim=1)  # (B,)

        outputs = torch.zeros(B, 512, 4, 4, device=x.device)

        # 🚦 expert execution
        for i in range(self.num_experts):
            mask = idx == i
            if mask.sum() == 0:
                continue

            out = self.experts[i](x[mask], return_features=True)
            outputs[mask] = out

        # final classification
        out = outputs.mean(dim=(2, 3))  # GAP
        out = self.head(out)

        return out


def model_factory(num_classes: int = 10) -> nn.Module:
    return ResNet18()
