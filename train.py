from torch.utils.data import random_split, Subset
from configs.hyperparameters import HyperParameters
from src.data.dataset import get_training_set
from src.data.dataloader import get_dataloader
from src.transform import *
from src.core.trainer import Trainer
from src.models import c2d, c2da, c2dp
from src.utils import *
import warnings


def main():
    warnings.filterwarnings("ignore")
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


    reduce_factor = 1
    reduced_size = int(len(training_data) * reduce_factor)
    indices = torch.randperm(len(training_data)).tolist()[:reduced_size]
    training_data = Subset(training_data, indices)

    train_size = int(0.8 * len(training_data))
    val_size = len(training_data) - train_size

    train_dataset, val_dataset = random_split(training_data, [train_size, val_size])

    train_dataloader = get_dataloader(opt, train_dataset)
    val_dataloader = get_dataloader(opt, val_dataset)

    model = c2dp.get_model(num_classes=opt.num_classes)

    model.calculate_parameter_size()
    trainer = Trainer(opt, model, train_dataloader, val_dataloader)

    header = ['Epoch', 'Train_Loss', 'Train_Acc@1', 'Train_Acc@5', 'Val_Loss', 'Val_Acc@1', 'Val_Acc@5', 'Val_Acc_Class']
    logger = Logger(header)

    for epoch in range(opt.epochs):
        trainer.train_epoch(epoch, logger)
        trainer.validate(epoch, logger)

if __name__ == "__main__":
    main()
