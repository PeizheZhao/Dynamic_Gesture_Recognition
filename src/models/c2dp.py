import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast
import torchvision.models as models
import torch.nn.functional as F
from .base_model import BaseModel

class PretrainedVideoCNN(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.bn = nn.BatchNorm2d(512)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.3)  # 新增特征后Dropout

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.bn(x)
        x = self.gelu(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)  # 应用Dropout
        return x

class FusionModel(BaseModel):
    def __init__(self, num_classes):
        super().__init__()
        # 视频处理分支
        self.video_cnn = PretrainedVideoCNN()
        self.video_lstm = nn.LSTM(1024, 512, batch_first=True, num_layers=2, 
                                bidirectional=True, dropout=0.3)  # 新增LSTM层间Dropout
        
        # 传感器分支
        self.sensor_net = nn.Sequential(
            nn.Conv1d(80, 192, 5, padding=2),
            nn.BatchNorm1d(192),
            nn.GELU(),
            nn.Dropout(0.2),  # 新增卷积后Dropout
            nn.Conv1d(192, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten()
        )
        self.sensor_lstm = nn.LSTM(256, 256, batch_first=True, num_layers=1, bidirectional=True)
        
        # 融合分类
        self.fusion_attention = nn.Sequential(
            nn.Linear(1024 + 512, 768),
            nn.GELU(),
            nn.Dropout(0.4),  # 新增注意力Dropout
            nn.Linear(768, 1024 + 512),
            nn.Sigmoid()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(1024 + 512, 768),
            nn.GELU(),
            nn.Dropout(0.6),  # 调高Dropout率
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(0.6),
            nn.Linear(512, num_classes)
        )

    def forward(self, c1, c2, sensor):
        with autocast(enabled=True):
            # 视频处理
            batch_size, _, timesteps, H, W = c1.shape
            
            def process_camera(view):
                view = view.permute(0, 2, 1, 3, 4).contiguous()
                features = checkpoint(self.video_cnn, view.view(-1, 3, H, W))
                return features.view(batch_size, timesteps, -1)
            
            c1_features = process_camera(c1)
            c2_features = process_camera(c2)
            
            combined_video = torch.cat([c1_features, c2_features], dim=2)
            video_out, _ = checkpoint(self.video_lstm, combined_video)
            video_feat = video_out[:, -1, :]
            video_feat = F.dropout(video_feat, 0.3, training=self.training)  # 新增输出后Dropout

            # 传感器处理
            sensor_feat = self.sensor_net(sensor)
            sensor_feat = sensor_feat.unsqueeze(1).repeat(1, 10, 1)
            sensor_out, _ = checkpoint(self.sensor_lstm, sensor_feat)
            sensor_feat = sensor_out[:, -1, :]
            sensor_feat = F.dropout(sensor_feat, 0.3, training=self.training)  # 新增输出后Dropout

            # 融合分类
            combined = torch.cat([video_feat, sensor_feat], dim=1)
            combined = F.dropout(combined, 0.3, training=self.training)  # 新增融合前Dropout
            
            attention = self.fusion_attention(combined)
            attended_features = combined * attention
            attended_features = F.dropout(attended_features, 0.5, training=self.training)  # 新增最终Dropout
            
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