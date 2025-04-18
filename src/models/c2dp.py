import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights
from .base_model import BaseModel


class PretrainedVideoViT(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载预训练 ViT-B/16
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        # 把原分类 heads 替换成 Identity，forward(x) 就直接返回 [CLS] token 的 768 维特征
        self.vit.heads = nn.Identity()
        # 简单再加个激活 + Dropout
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x: (B, 3, H, W)
        feats = self.vit(x)       # -> (B, 768), CLS token 输出，无分类 head
        feats = self.gelu(feats)
        feats = self.dropout(feats)
        return feats              # -> (B, 768)


class FusionModel(BaseModel):
    def __init__(self, num_classes):
        super().__init__()
        # 视频分支：两路摄像头
        self.video_cnn = PretrainedVideoViT()
        self.video_lstm = nn.LSTM(
            input_size=768 * 2,       # 两路 CLS 特征拼接 -> 1536
            hidden_size=768,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # 传感器分支保持不变
        self.sensor_lstm = nn.LSTM(
            input_size=9,  # 修正传感器特征维度
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # 融合注意力：视频 1536 + 传感器 512 = 2048
        self.fusion_attention = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 2048),
            nn.Sigmoid()
        )

        # 最终分类头
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Dropout(0.6),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.6),
            nn.Linear(512, num_classes)
        )

    def forward(self, c1, c2, sensor):
        with autocast(enabled=True):
            B, _, T, H, W = c1.shape

            def process_camera(view):
                # view: (B, 3, T, H, W) -> (B, T, 3, H, W) -> (B*T, 3, H, W)
                v = view.permute(0, 2, 1, 3, 4).reshape(-1, 3, H, W)
                feats = checkpoint(self.video_cnn, v)   # -> (B*T, 768)
                return feats.view(B, T, -1)            # -> (B, T, 768)

            # 处理两路摄像头序列
            f1 = process_camera(c1)
            f2 = process_camera(c2)
            video_seq = torch.cat([f1, f2], dim=2)      # -> (B, T, 1536)

            video_out, _ = checkpoint(self.video_lstm, video_seq)
            video_feat = video_out[:, -1, :]            # -> (B, 1536)
            video_feat = F.dropout(video_feat, 0.3, self.training)

            # 处理传感器数据
            sensor_out, _ = checkpoint(self.sensor_lstm, sensor)  # 修正传感器数据处理
            sensor_feat = sensor_out[:, -1, :]          # -> (B, 512)
            sensor_feat = F.dropout(sensor_feat, 0.3, self.training)

            # 融合与注意力
            fused = torch.cat([video_feat, sensor_feat], dim=1)  # -> (B, 2048)
            fused = F.dropout(fused, 0.3, self.training)
            attn = self.fusion_attention(fused)                  # -> (B, 2048)
            fused = fused * attn
            fused = F.dropout(fused, 0.5, self.training)

            # 最终分类
            return self.fc(fused.float())


def get_model(**kwargs):
    return FusionModel(**kwargs)


if __name__ == "__main__":
    # 简单验证
    c1 = torch.randn(16, 3, 8, 224, 224)
    c2 = torch.randn(16, 3, 8, 224, 224)
    sensor = torch.randn(16, 80, 9)
    model = FusionModel(num_classes=30)
    out = model(c1, c2, sensor)
    print("Output shape:", out.shape)  # 应为 (16, 30)