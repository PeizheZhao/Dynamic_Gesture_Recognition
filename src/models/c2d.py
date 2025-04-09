import torch
import torch.nn as nn
from .base_model import BaseModel
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast

class VideoCNN(nn.Module):
    """处理单帧的2D CNN"""
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112x112
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 56x56
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # 全局池化
        )
    
    def forward(self, x):
        with autocast(enabled=True):  # 启用混合精度
            return self.cnn(x).flatten(1)

class FusionModel(BaseModel):
    def __init__(self, num_classes):
        super().__init__()
        # 视频处理分支
        self.video_cnn = VideoCNN()
        self.video_lstm = nn.LSTM(512, 256, batch_first=True)  # 双摄像头拼接后512
        
        # 传感器处理分支
        self.sensor_lstm = nn.LSTM(80, 128, batch_first=True)
        
        # 融合分类层
        self.fc = nn.Sequential(
            nn.Linear(256 + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, c1, c2, sensor):
        with autocast(enabled=True):  # 整个前向传播启用混合精度
            # 处理双摄像头视频流 -------------------------------------------------
            batch_size, _, timesteps, H, W = c1.shape
            
            # 使用检查点处理视频帧(节省内存)
            def process_camera(view):
                view = view.permute(0, 2, 1, 3, 4).contiguous()
                features = checkpoint(self.video_cnn, view.view(-1, 3, H, W))  # 使用检查点
                return features.view(batch_size, timesteps, -1)
            
            c1_features = process_camera(c1)  # (16, 8, 256)
            c2_features = process_camera(c2)  # (16, 8, 256)

            # 合并双摄像头特征
            combined_video = torch.cat([c1_features, c2_features], dim=2)  # (16, 8, 512)
            
            # 视频LSTM (使用检查点)
            video_out, _ = checkpoint(self.video_lstm, combined_video)
            video_feat = video_out[:, -1, :]  # 取最后一个时间步 (16, 256)

            # 处理传感器数据 -----------------------------------------------------
            # 输入形状转换: (batch, features, time) -> (batch, time, features)
            sensor = sensor.permute(0, 2, 1)  # (16, 10, 80)
            
            # 传感器LSTM (使用检查点)
            sensor_out, _ = checkpoint(self.sensor_lstm, sensor)
            sensor_feat = sensor_out[:, -1, :]  # (16, 128)

            # 特征融合 ----------------------------------------------------------
            combined = torch.cat([video_feat, sensor_feat], dim=1)  # (16, 384)
            
            # 最终分类层保持float32精度以确保稳定性
            with autocast(enabled=False):
                combined = combined.float()  # 确保输入为float32
                return self.fc(combined)

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
    sensor = torch.randn(16, 80, 10)
    
    model = FusionModel(num_classes=30)
    output = model(c1, c2, sensor)
    print(f"Output shape: {output.shape}")  # 应该得到 torch.Size([16, 30])