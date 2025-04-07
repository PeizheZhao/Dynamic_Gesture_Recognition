from configs.hyperparameters import HyperParameters
from src.data.dataset import get_training_set
from src.data.dataloader import get_dataloader
from src.transform.spatial_transforms import *
from src.transform.temporal_transforms import *
from src.transform.target_transforms import ClassLabel, VideoID
from src.transform.target_transforms import Compose as TargetCompose


opt = HyperParameters(
    "/Users/zhaopeizhe/Documents/Share_Space/Dynamic_Gesture/Code/Project/configs/train_config.yaml"
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

dataloader = get_dataloader(opt, training_data)

if __name__ == '__main__':
    for batch_idx,(c1,c2,s,labels) in enumerate(dataloader):
        if batch_idx >= 3:
            break
        print(f"Batch {batch_idx + 1}")
        print(f"c1 shape: {c1.shape}")
        print(f"c2 shape: {c2.shape}")
        print(f"sensor shape: {s.shape}")
        print(f"Labels: {labels}")
        print("-" * 20)
