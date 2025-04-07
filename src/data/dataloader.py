from typing import List, Tuple
import torch
from torch import Tensor
from torch.utils.data import DataLoader


def collate_fn(batch):
    tensor1_batch = torch.stack([item[0] for item in batch])
    tensor2_batch = torch.stack([item[1] for item in batch])
    tensor3_batch = torch.stack([item[2] for item in batch])
    labels_batch = torch.tensor([int(item[3]) for item in batch])
    return [tensor1_batch, tensor2_batch, tensor3_batch], labels_batch


def get_dataloader(opt, dataset):
    return DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

if __name__=="__main__":
    import sys
    import os
    project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    sys.path.append(project_root)
    from configs.hyperparameters import HyperParameters
    from src.data.dataset import get_training_set
    from src.data.dataloader import get_dataloader
    from src.transform.spatial_transforms import *
    from src.transform.temporal_transforms import *
    from src.transform.target_transforms import ClassLabel, VideoID
    from src.transform.target_transforms import Compose as TargetCompose
    from torch.utils.data import random_split


    opt = HyperParameters(
        "configs/train_config.yaml"
    )

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    if not opt.no_train:
        assert opt.train_crop in ["random", "corner", "center"]
        if opt.train_crop == "random":
            crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == "corner":
            crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == "center":
            crop_method = MultiScaleCornerCrop(
                opt.scales, opt.sample_size, crop_positions=["c"]
            )

    spatial_transform = Compose(
        [
            crop_method,
            ToTensor(opt.norm_value),
            norm_method,
        ]
    )

    temporal_transform = TemporalCenterCrop(opt.sample_duration, opt.downsample)

    target_transform = ClassLabel()

    training_data = get_training_set(
        opt, spatial_transform, temporal_transform, target_transform
    )

    train_size = int(0.8 * len(training_data))  
    val_size = len(training_data) - train_size   

    train_dataset, val_dataset = random_split(training_data, [train_size, val_size])

    train_dataloader = get_dataloader(opt, train_dataset)
    val_dataloader = get_dataloader(opt, val_dataset)

    for batch_idx,(inputs, labels) in enumerate(train_dataloader):
        if batch_idx >= 3:
            break
        print(f"Batch {batch_idx + 1}")
        print(f"c1 shape: {inputs[0].shape}")
        print(f"c2 shape: {inputs[1].shape}")
        print(f"sensor shape: {inputs[2].shape}")
        print(f"Labels: {labels}")
        print("-" * 20)