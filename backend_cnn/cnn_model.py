import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock1D(nn.Module):
    """
    Squeeze-and-Excitation channel attention block for 1D feature representations.
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, max(channels // reduction, 8)),
            nn.GELU(),
            nn.Linear(max(channels // reduction, 8), channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(x).unsqueeze(-1)
        return x * w


class ResidualBlock1D(nn.Module):
    """
    1D Residual Block with GroupNorm, GELU, Squeeze-and-Excitation attention, and Dropout.
    """
    def __init__(self, in_c, out_c, stride=1, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, kernel_size=5, stride=stride, padding=2, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=min(8, out_c), num_channels=out_c)
        self.act1 = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_c, out_c, kernel_size=5, stride=1, padding=2, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=min(8, out_c), num_channels=out_c)
        self.se = SEBlock1D(out_c)
        self.act2 = nn.GELU()

        self.shortcut = nn.Sequential()
        if in_c != out_c or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=min(8, out_c), num_channels=out_c)
            )

    def forward(self, x):
        res = self.shortcut(x)
        out = self.act1(self.gn1(self.conv1(x)))
        out = self.drop(out)
        out = self.gn2(self.conv2(out))
        out = self.se(out)
        out = self.act2(out + res)
        return out


class EmotionCNN(nn.Module):
    """
    Deep 1D Residual CNN with Squeeze-and-Excitation Attention for Speech Emotion Recognition.
    Accepts input shape (Batch, 526) or (Batch, 1, 526).
    """
    def __init__(self, in_features=526, num_classes=8):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes

        # 1D Convolutional Stem
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.MaxPool1d(2)
        )

        self.layer1 = ResidualBlock1D(64, 128, stride=2, dropout=0.2)
        self.layer2 = ResidualBlock1D(128, 256, stride=2, dropout=0.25)
        self.layer3 = ResidualBlock1D(256, 256, stride=2, dropout=0.3)

        self.pool = nn.AdaptiveAvgPool1d(1)

        # Dense Classifier with LayerNorm
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out).flatten(1)
        out = self.classifier(out)
        return out


if __name__ == "__main__":
    model = EmotionCNN(in_features=526, num_classes=8)
    dummy = torch.randn(4, 1, 526)
    out = model(dummy)
    print("EmotionCNN Output Shape:", out.shape)