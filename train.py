from torch.utils.data import random_split
from configs.hyperparameters import HyperParameters
from src.data.dataset import get_training_set
from src.data.dataloader import get_dataloader
from src.transform import *
from src.core.trainer import Trainer
from src.models import c3d


def main():
    opt = HyperParameters("configs/train_config.yaml")

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

    model = c3d.get_model(sample_size=224, sample_duration=8, num_classes=32)

    trainer = Trainer(opt, model, train_dataloader, val_dataloader)
    for epoch in range(opt.epochs):
        trainer.train_epoch(epoch)
        trainer.validate(epoch)


if __name__ == "__main__":
    main()
