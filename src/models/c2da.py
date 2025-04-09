import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast
from .base_model import BaseModel

class ChannelAttention(nn.Module):
    """通道注意力机制(精简版)"""
    def __init__(self, channels, reduction=16):  # reduction从16改为8
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).squeeze())
        max_out = self.fc(self.max_pool(x).squeeze())
        out = self.sigmoid(avg_out + max_out).unsqueeze(-1).unsqueeze(-1)
        return x * out

class ResidualBlock(nn.Module):
    """增强的残差块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()
        
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None
    def forward(self, x):
        residual = x
        out = self.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.gelu(out)

class SpatialAttention(nn.Module):
    """空间注意力机制"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))
        return x * out

class EfficientVideoCNN(nn.Module):
    """精简版的2D CNN特征提取器"""
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            # 第一卷积层(缩小通道数)
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),  # 112x112
            
            # 残差块组(减少块数量和通道数)
            ResidualBlock(64, 128, stride=1),
            ResidualBlock(128, 128, stride=1),
            
            # 中间层
            nn.Conv2d(128, 192, 3, padding=1),
            nn.BatchNorm2d(192),
            nn.GELU(),
            SpatialAttention(),
            nn.MaxPool2d(2),  # 56x56
            
            # 通道注意力模块
            ChannelAttention(192),
            
            # 深度可分离卷积层(减少分组数)
            nn.Conv2d(192, 256, 3, padding=1, groups=16),
            nn.BatchNorm2d(256),
            nn.GELU(),
            
            # 残差块(减少数量)
            ResidualBlock(256, 256),
            
            # 最终聚合
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 512, 1),  # 输出维度减半
            nn.BatchNorm2d(512),
            nn.GELU()
        )
    
    def forward(self, x):
        with autocast(enabled=True):
            return self.cnn(x).flatten(1)

class FusionModel(BaseModel):
    def __init__(self, num_classes):
        super().__init__()
        # 精简的视频处理分支
        self.video_cnn = EfficientVideoCNN()
        self.video_lstm = nn.LSTM(1024, 512, batch_first=True, num_layers=2, bidirectional=True)  # 维度减半
        
        # 精简的传感器处理分支
        self.sensor_net = nn.Sequential(
            nn.Conv1d(80, 192, 5, padding=2),  # 通道数减少
            nn.BatchNorm1d(192),
            nn.GELU(),
            nn.Conv1d(192, 256, 3, padding=1),  # 减少层数
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten()
        )
        self.sensor_lstm = nn.LSTM(256, 256, batch_first=True, num_layers=1, bidirectional=True)  # 单层LSTM
        
        # 精简的融合分类层
        self.fusion_attention = nn.Sequential(
            nn.Linear(1024 + 512, 768),  # 缩小维度
            nn.GELU(),
            nn.Linear(768, 1024 + 512),
            nn.Sigmoid()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(1024 + 512, 768),  # 缩小全连接层
            nn.GELU(),
            nn.Dropout(0.55),  # 降低dropout率
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(0.55),
            nn.Linear(512, num_classes)
        )

    def forward(self, c1, c2, sensor):
        with autocast(enabled=True):
            # 处理双摄像头视频流 --------------------------------
            batch_size, _, timesteps, H, W = c1.shape
            
            def process_camera(view):
                view = view.permute(0, 2, 1, 3, 4).contiguous()
                features = checkpoint(self.video_cnn, view.view(-1, 3, H, W))
                return features.view(batch_size, timesteps, -1)
            
            c1_features = process_camera(c1)  # (16, 8, 512)
            c2_features = process_camera(c2)  # (16, 8, 512)
            
            # 合并双摄像头特征
            combined_video = torch.cat([c1_features, c2_features], dim=2)  # (16, 8, 1024)
            
            # 双向LSTM处理时序
            video_out, _ = checkpoint(self.video_lstm, combined_video)
            video_feat = video_out[:, -1, :]  # (16, 1024)

            # 处理传感器数据 -------------------------------------------------
            # 先用CNN提取局部特征
            sensor_feat = self.sensor_net(sensor)  # (16, 256)
            sensor_feat = sensor_feat.unsqueeze(1).repeat(1, 10, 1)  # (16, 10, 256)
            
            # 双向LSTM处理时序
            sensor_out, _ = checkpoint(self.sensor_lstm, sensor_feat)
            sensor_feat = sensor_out[:, -1, :]  # (16, 512)

            # 特征融合 ------------------------------------------------------
            combined = torch.cat([video_feat, sensor_feat], dim=1)  # (16, 1536)
            
            # 特征注意力融合
            attention = self.fusion_attention(combined)
            attended_features = combined * attention
            
            return self.fc(attended_features.float())
        
def get_model(**kwargs):
    """
    Returns the model.
    """
    model = FusionModel(**kwargs)
    return model

# 验证网络结构
if __name__ == "__main__":
    # 模拟输入数据
    c1 = torch.randn(16, 3, 8, 224, 224)
    c2 = torch.randn(16, 3, 8, 224, 224)
    sensor = torch.randn(16, 80, 9)
    
    model = FusionModel(num_classes=30)
    output = model(c1, c2, sensor)
    print(f"Output shape: {output.shape}")  