# Dynamic Gesture Recognition Project / 动态手势识别项目

## 🌍 English Version

### Overview

This project implements a deep learning method for dynamic gesture recognition using the **Dynamic Gesture Dataset**. The dataset consists of refined gesture classes and is designed for training and evaluating gesture recognition models.

### Directory Structure

```
.
├── README.md
├── __init__.py
├── checkpoints
├── configs
│   ├── hyperparameters.py
│   └── train_config.yaml
├── guide.md
├── logs
├── src
│   ├── __init__.py
│   ├── __pycache__
│   ├── core
│   ├── data
│   ├── experiments
│   ├── models
│   ├── transform
│   └── utils
├── test.py
└── train.py
```

### Changelog



### Installation

To install the dependencies, run:

```bash
pip install -r requirements.txt
```

### Usage

1. **Prepare the dataset**: Follow the instructions in `docs/GUIDE_GIT.md` to prepare the dataset.
2. **Run the training script**: Use the following command to start training:

```bash
python src/train.py --config config.yaml
```

### Contributing

Contributions are welcome! Please follow the standard GitHub workflow for contributions.

### License

This project is licensed under the MIT License.

For more information, please refer to the documentation in the `docs` folder.

---

## 中文版本

### 项目概述

该项目实现了一种用于动态手势识别的深度学习方法，使用**动态手势数据集**。该数据集由优化的手势类别组成，旨在用于手势识别模型的训练和评估。

### 目录结构

```
.
├── README.md
├── __init__.py
├── checkpoints
├── configs
│   ├── hyperparameters.py
│   └── train_config.yaml
├── guide.md
├── logs
├── src
│   ├── __init__.py
│   ├── __pycache__
│   ├── core
│   ├── data
│   ├── experiments
│   ├── models
│   ├── transform
│   └── utils
├── test.py
└── train.py
```

### 更新日志


### 安装

安装依赖项，请运行：

```bash
pip install -r requirements.txt
```

### 使用

1. **准备数据集**: 按照 `docs/GUIDE_GIT.md` 中的说明准备数据集。
2. **运行训练脚本**: 使用以下命令开始训练：

```bash
python src/train.py --config config.yaml
```

### 贡献

欢迎贡献！请遵循标准的 GitHub 工作流程进行贡献。

### 许可证

此项目遵循 MIT 许可证。

有关更多信息，请参阅 `docs` 文件夹中的文档。

