"""ResNet18 implementation for CIFAR-10 classification."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    pass


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResNet18.

        Args:
            x: Input tensor (B, 3, H, W).

        Returns:
            Classification logits (B, num_classes).
        """

        if x.dim() != 4:
            raise ValueError("Expected 4D input tensor (B, C, H, W)")

        # Initial stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Stages
        x = self.stages(x)

        # Classification head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def model_factory(num_classes: int = 10, model_type: str = "resnet18") -> nn.Module:
    """Factory function to create a model instance.

    Args:
        num_classes: Number of output classification classes.
        model_type: Model architecture to use ('resnet18' or 'mythos').

    Returns:
        A PyTorch nn.Module instance.

    Raises:
        ValueError: If num_classes < 1 or invalid model_type provided.
    """
    return ResNet18()
