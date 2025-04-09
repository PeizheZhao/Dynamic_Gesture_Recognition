from .base_model import BaseModel
import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.utils.checkpoint import checkpoint


class C3D(BaseModel):
    def __init__(self, sample_size=224, sample_duration=8, num_classes=600):
        super(C3D, self).__init__()

        # 更紧凑的特征提取器
        self.feature_extractor = nn.Sequential(
            # group1
            nn.Conv3d(3, 32, kernel_size=3, padding=1),  # 减少输出通道数
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2)),
            # group2
            nn.Conv3d(32, 64, kernel_size=3, padding=1),  # 减少输出通道数
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            # group3
            nn.Conv3d(64, 128, kernel_size=3, padding=1),  # 减少输出通道数
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),  # 减少输出通道数
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            # group4-5
            # 可以考虑进一步减少层数
        )

        # 动态计算特征维度
        dummy = torch.zeros(1, 3, sample_duration, sample_size, sample_size)
        with torch.no_grad():
            dummy_out = self.feature_extractor(dummy)
        self.feature_dim = dummy_out.view(1, -1).size(1)

        # 分支网络
        self.fc_video = nn.Linear(self.feature_dim, 512)  # 降低输出尺寸
        self.sensor_net = nn.Sequential(
            nn.Conv1d(in_channels=80, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Add this to reduce temporal dimension
            nn.Flatten(),
            nn.Linear(128, 128),  # Add this to ensure exact output size
        )
        self.fc_fusion = nn.Sequential(
            nn.Linear(512 * 2 + 128, 512),  # 更紧凑的融合
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, c1, c2, sensor):
        with torch.amp.autocast("cuda"):
            # 视频分支（共享权重+检查点）
            feat1 = checkpoint(self.feature_extractor, c1).flatten(1)
            feat1 = self.fc_video(feat1)

            feat2 = checkpoint(self.feature_extractor, c2).flatten(1)
            feat2 = self.fc_video(feat2)

            # 传感器分支
            sensor_feat = self.sensor_net(sensor)

            # 融合
            out = self.fc_fusion(torch.cat([feat1, feat2, sensor_feat], dim=1))
        return out


def get_model(**kwargs):
    """
    Returns the model.
    """
    model = C3D(**kwargs)
    return model


if __name__ == "__main__":
    model = get_model(sample_size=224, sample_duration=8, num_classes=32)
    c1 = torch.randn(16, 3, 8, 224, 224)
    c2 = torch.randn(16, 3, 8, 224, 224)
    sensor = torch.randn(16, 80, 10)
    output = model(c1, c2, sensor)
    print("Output shape:", output.shape)
