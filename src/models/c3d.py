from .base_model import BaseModel
import torch.nn as nn
import torch
from torch.autograd import Variable

class C3D(BaseModel):
    def __init__(self, sample_size=224, sample_duration=8, num_classes=600):
        super(C3D, self).__init__()
        
        # 共享的视频特征提取网络
        self.feature_extractor = nn.Sequential(
            # group1
            nn.Conv3d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2,2,2), stride=(1,2,2)),  # 时间维度只轻微下降
            # group2
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2)),
            # group3
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2)),
            # group4 —— 修改时间维度池化，避免过度下采样
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)),  # 时间维度不做下采样
            # group5 —— 同样在时间上不做下采样
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2), padding=(0,1,1))
        )
        
        # 动态计算视频分支特征的维度
        dummy = torch.zeros(1, 3, sample_duration, sample_size, sample_size)
        with torch.no_grad():
            dummy_out = self.feature_extractor(dummy)
        self.feature_dim = dummy_out.view(1, -1).size(1)
        
        # 视频分支全连接处理
        self.fc_video = nn.Sequential(
            nn.Linear(self.feature_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        
        # 传感器分支：假设传感器输入形状为 [B, 80, 10]
        self.sensor_net = nn.Sequential(
            nn.Flatten(),  # 将 [B,80,10] 变为 [B,800]
            nn.Linear(80 * 10, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        
        # 融合两个视频分支和传感器特征
        # 两路视频各 2048 维，传感器为 256 维，总共 2048*2 + 256
        self.fc_fusion = nn.Sequential(
            nn.Linear(2048 * 2 + 256, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes)
        )
    
    def forward(self, c1, c2, sensor):
        # 处理视频分支1
        feat1 = self.feature_extractor(c1)
        feat1 = feat1.view(feat1.size(0), -1)
        feat1 = self.fc_video(feat1)
        
        # 处理视频分支2
        feat2 = self.feature_extractor(c2)
        feat2 = feat2.view(feat2.size(0), -1)
        feat2 = self.fc_video(feat2)
        
        # 处理传感器分支
        sensor_feat = self.sensor_net(sensor)
        
        # 融合所有特征
        fusion = torch.cat([feat1, feat2, sensor_feat], dim=1)
        out = self.fc_fusion(fusion)
        return out

def get_model(**kwargs):
    """
    Returns the model.
    """
    model = C3D(**kwargs)
    return model


if __name__ == '__main__':
    model = get_model(sample_size=224, sample_duration=8, num_classes=32)
    c1 = torch.randn(16, 3, 8, 224, 224)
    c2 = torch.randn(16, 3, 8, 224, 224)
    sensor = torch.randn(16, 80, 10)
    output = model(c1, c2, sensor)
    print("Output shape:", output.shape) 