以下是一个标准的深度学习项目文件结构与代码组织方案，帮助你实现高可维护性的代码架构。假设你的项目名为`DLProject`：

### 文件目录结构
```bash
DLProject/
│
├── configs/               # 配置文件目录
│   ├── train_config.yaml  # 训练超参数配置
│   └── model_config.yaml  # 模型结构配置
│
├── data/                  # 数据相关
│   ├── raw/               # 原始数据
│   ├── processed/         # 预处理后的数据
│   └── README.md          # 数据说明文档
│
├── src/                   # 主代码目录
│   ├── __init__.py
│   ├── data/              # 数据加载与预处理
│   │   ├── __init__.py
│   │   ├── datasets.py    # 自定义Dataset类
│   │   └── dataloaders.py # 数据加载器生成
│   │
│   ├── models/            # 模型定义
│   │   ├── __init__.py
│   │   ├── base_model.py  # 抽象基类
│   │   ├── vision/        # 计算机视觉模型
│   │   │   ├── resnet.py
│   │   │   └── transformer.py
│   │   └── nlp/           # 自然语言处理模型
│   │       └── bert.py
│   │
│   ├── core/              # 核心训练逻辑
│   │   ├── trainer.py     # 训练器类
│   │   └── tester.py      # 测试器类
│   │
│   ├── utils/             # 工具函数
│   │   ├── logger.py      # 日志记录
│   │   ├── metrics.py     # 评估指标
│   │   └── visualize.py   # 可视化工具
│   │
│   └── experiments/       # 实验脚本（可选）
│       └── exp001.py
│
├── checkpoints/           # 模型保存目录
├── logs/                  # 训练日志目录
├── requirements.txt       # 依赖清单
├── train.py               # 主训练脚本
├── test.py                # 主测试脚本
└── README.md              # 项目文档
```

---

### 关键文件说明

#### 1. 数据模块 (`src/data/`)
```python
# datasets.py
from torch.utils.data import Dataset

class ClassificationDataset(Dataset):
    """自定义数据集示例"""
    def __init__(self, data_path, transform=None):
        self.data = ...  # 加载数据
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# dataloaders.py
from torch.utils.data import DataLoader

def create_dataloader(dataset, batch_size=32, shuffle=True):
    """创建标准数据加载器"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )
```

#### 2. 模型模块 (`src/models/`)
```python
# base_model.py
import torch.nn as nn

class BaseModel(nn.Module):
    """模型抽象基类"""
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        raise NotImplementedError
        
    def get_optimizer(self, lr):
        return torch.optim.Adam(self.parameters(), lr=lr)

# vision/resnet.py
from .base_model import BaseModel

class ResNet(BaseModel):
    """具体模型实现"""
    def __init__(self, num_classes=10):
        super().__init__()
        # 定义网络层
        
    def forward(self, x):
        # 前向传播逻辑
        return x
```

#### 3. 核心训练逻辑 (`src/core/trainer.py`)
```python
class Trainer:
    """训练器封装"""
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.optimizer = model.get_optimizer(config['lr'])
        self.criterion = nn.CrossEntropyLoss()
        self.device = config['device']
        
    def train_epoch(self):
        self.model.train()
        for batch in self.train_loader:
            inputs, labels = batch
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
    
    def validate(self):
        self.model.eval()
        with torch.no_grad():
            # 验证逻辑
            pass
            
    def save_checkpoint(self, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, f'checkpoints/model_epoch{epoch}.pth')
```

#### 4. 主训练脚本 (`train.py`)
```python
import yaml
from src.data import ClassificationDataset, create_dataloader
from src.models.vision import ResNet
from src.core import Trainer

def main():
    # 加载配置
    with open('configs/train_config.yaml') as f:
        config = yaml.safe_load(f)
    
    # 初始化数据
    train_data = ClassificationDataset(...)
    train_loader = create_dataloader(train_data)
    
    # 创建模型
    model = ResNet(num_classes=10)
    
    # 初始化训练器
    trainer = Trainer(model, train_loader, None, config)
    
    # 训练循环
    for epoch in range(config['epochs']):
        trainer.train_epoch()
        if epoch % 5 == 0:
            trainer.save_checkpoint(epoch)

if __name__ == '__main__':
    main()
```

---

### 依赖管理
1. 创建`requirements.txt`：
```txt
torch>=2.0.1
torchvision
numpy
tqdm
PyYAML
tensorboard  # 可选，用于可视化
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

---

### 最佳实践建议
1. 配置分离：将超参数、模型结构等配置与代码分离，使用YAML文件管理
2. 模块化设计：每个Python文件只做一件事，保持单一职责原则
3. 日志记录：在`utils/logger.py`中实现统一的日志记录
4. 单元测试：为关键模块编写测试用例（可添加tests目录）
5. 类型提示：推荐使用Python类型提示提高可读性
6. 分布式训练：在Trainer类中预留DDP分布式训练接口

### 1. 工具模块 (`src/utils/`)

#### 1.1 日志记录 (`logger.py`)
```python
import logging
from pathlib import Path
from datetime import datetime

class ExperimentLogger:
    """统一日志记录器（单例模式）"""
    _instance = None
    
    def __new__(cls, log_dir="logs"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            Path(log_dir).mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = Path(log_dir) / f"exp_{timestamp}.log"
            
            # 配置日志系统
            logger = logging.getLogger("DLProject")
            logger.setLevel(logging.DEBUG)
            
            # 文件Handler
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            
            # 控制台Handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter('%(levelname)s: %(message)s')
            console_handler.setFormatter(console_formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            cls._instance.logger = logger
            
        return cls._instance

    @classmethod
    def info(cls, message):
        cls().logger.info(message)
    
    @classmethod
    def warning(cls, message):
        cls().logger.warning(message)
```

#### 1.2 评估指标 (`metrics.py`)
```python
import torch
from sklearn.metrics import f1_score, roc_auc_score

class BaseMetrics:
    """指标计算基类"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.preds = []
        self.targets = []
    
    def update(self, outputs, targets):
        raise NotImplementedError
        
    def compute(self):
        raise NotImplementedError

class ClassificationMetrics(BaseMetrics):
    """分类任务指标套件"""
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
    def update(self, outputs, targets):
        _, preds = torch.max(outputs, 1)
        self.preds.extend(preds.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
    
    def compute(self):
        return {
            "accuracy": sum(p == t for p, t in zip(self.preds, self.targets)) / len(self.targets),
            "f1_macro": f1_score(self.targets, self.preds, average='macro'),
            "auc_ovo": roc_auc_score(self.targets, self.preds, multi_class='ovo')
        }

class RegressionMetrics(BaseMetrics):
    """回归任务指标套件"""
    def update(self, outputs, targets):
        self.preds.extend(outputs.view(-1).cpu().numpy())
        self.targets.extend(targets.view(-1).cpu().numpy())
        
    def compute(self):
        preds_tensor = torch.tensor(self.preds)
        targets_tensor = torch.tensor(self.targets)
        return {
            "mae": torch.mean(torch.abs(preds_tensor - targets_tensor)).item(),
            "mse": torch.mean((preds_tensor - targets_tensor)**2).item(),
            "r2": 1 - torch.var(preds_tensor - targets_tensor) / torch.var(targets_tensor)
        }
```

#### 1.3 可视化工具 (`visualize.py`)
```python
import matplotlib.pyplot as plt
import torchvision

def plot_feature_maps(feature_maps, ncols=8):
    """可视化卷积特征图"""
    plt.figure(figsize=(20, 5))
    for idx, feat_map in enumerate(feature_maps[0]):
        if idx >= ncols:
            break
        plt.subplot(1, ncols, idx+1)
        plt.imshow(feat_map.detach().cpu().numpy(), cmap='viridis')
        plt.axis('off')
    plt.tight_layout()
    return plt

def plot_data_samples(dataset, classes, num_samples=8):
    """展示数据样本"""
    plt.figure(figsize=(12, 6))
    indices = torch.randint(0, len(dataset), (num_samples,))
    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        plt.subplot(2, num_samples//2, i+1)
        plt.imshow(img.permute(1, 2, 0) if img.shape[0]==3 else img.squeeze(), cmap='gray')
        plt.title(f"Label: {classes[label]}")
        plt.axis('off')
    plt.tight_layout()
    return plt
```

---

### 2. 测试模块 (`src/core/tester.py`)
```python
class Tester:
    """模型测试框架"""
    def __init__(self, model, test_loader, metrics):
        self.model = model
        self.test_loader = test_loader
        self.metrics = metrics
        self.device = next(model.parameters()).device
    
    @torch.no_grad()
    def run_test(self):
        self.model.eval()
        self.metrics.reset()
        
        for inputs, targets in self.test_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            outputs = self.model(inputs)
            self.metrics.update(outputs, targets)
            
        return self.metrics.compute()
    
    def generate_report(self, results):
        """生成测试报告"""
        report = ["\n===== 测试报告 ====="]
        for metric, value in results.items():
            report.append(f"{metric:12s}: {value:.4f}")
        return "\n".join(report)
```

---

### 3. 主测试脚本 (`test.py`)
```python
import torch
import yaml
from src.models.vision import ResNet
from src.data import create_dataloader
from src.core import Tester
from src.utils import ClassificationMetrics

def main():
    # 加载配置
    with open('configs/model_config.yaml') as f:
        config = yaml.safe_load(f)
    
    # 初始化测试数据
    test_dataset = ...  # 加载测试数据集
    test_loader = create_dataloader(test_dataset, shuffle=False)
    
    # 加载模型
    model = ResNet(num_classes=config['num_classes'])
    model.load_state_dict(torch.load(config['model_path']))
    model.to(config['device'])
    
    # 初始化测试器
    metrics = ClassificationMetrics(num_classes=config['num_classes'])
    tester = Tester(model, test_loader, metrics)
    
    # 执行测试
    results = tester.run_test()
    print(tester.generate_report(results))

if __name__ == '__main__':
    main()
```

---

### 4. 实验脚本 (`src/experiments/exp001.py`)
```python
import argparse
from itertools import product
from src.utils import ExperimentLogger

def hyperparameter_search():
    """超参数搜索实验模板"""
    logger = ExperimentLogger()
    learning_rates = [1e-3, 3e-4]
    batch_sizes = [32, 64]
    
    for lr, bs in product(learning_rates, batch_sizes):
        logger.info(f"开始实验: lr={lr}, bs={bs}")
        # 此处插入训练流程
        # 记录实验结果
        logger.info(f"实验完成，准确率: 0.92")

def model_comparison():
    """模型对比实验模板"""
    models = ['resnet18', 'efficientnet', 'vit']
    for model_name in models:
        # 初始化不同模型
        # 执行训练验证流程
        # 记录性能指标
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_type', choices=['hparam', 'model'], required=True)
    args = parser.parse_args()
    
    if args.exp_type == 'hparam':
        hyperparameter_search()
    elif args.exp_type == 'model':
        model_comparison()
```

---

### 5. 配置文件示例 (`configs/`)

#### model_config.yaml
```yaml
# 模型架构配置
model_type: 'resnet50'
num_classes: 10
pretrained: True
input_size: [224, 224]
model_path: 'checkpoints/best_model.pth'
```

#### train_config.yaml
```yaml
# 训练超参数配置
device: 'cuda:0'
epochs: 100
batch_size: 64
learning_rate: 1e-3
optimizer: 'adam'
scheduler: 
  name: 'cosine'
  max_lr: 1e-3
  min_lr: 1e-5
early_stop_patience: 10
```

---

### 关键技术点说明

1. **日志系统**：
• 采用单例模式确保全局唯一日志实例
• 自动生成带时间戳的日志文件
• 同时输出到文件和控制台

2. **指标计算**：
• 支持分类与回归任务的常见指标
• 采用增量式计算节省内存
• 统一接口方便扩展新指标

3. **可视化工具**：
• 特征图可视化帮助理解模型工作原理
• 数据样本展示验证预处理正确性
• 返回plt对象允许进一步定制

4. **测试框架**：
• 支持任意指标组合
• 自动生成可读报告
• 与设备无关的设计

5. **实验管理**：
• 提供超参数搜索模板
• 支持模型对比实验
• 可通过命令行参数选择实验类型

---

### 完整工作流程示例

1. **训练模型**：
```bash
python train.py --config configs/train_config.yaml
```

2. **执行测试**：
```bash
python test.py --model_path checkpoints/best_model.pth
```

3. **运行实验**：
```bash
python -m src.experiments.exp001 --exp_type hparam
```

4. **查看日志**：
```bash
tail -f logs/exp_20230801_1430.log
```

5. **可视化分析**：
```python
from src.utils.visualize import plot_feature_maps
feat_maps = model.conv_layers(input_image)
plot_feature_maps(feat_maps).show()
```

这种架构设计可以支持从快速原型开发到生产部署的全流程需求，建议根据具体任务需求调整数据加载策略、模型架构和评估指标。对于大型项目，可考虑添加DVC进行数据版本控制，使用MLflow进行实验跟踪。