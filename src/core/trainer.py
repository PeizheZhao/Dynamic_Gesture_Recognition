import torch
import torch.nn as nn
import time
from tqdm import tqdm
from ..utils import *


class Trainer:
    def __init__(self, opt, model, train_loader, val_loader):
        self.model = model
        self.optimizer = model.get_optimizer(lr = opt.lr, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.opt = opt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train_epoch(self, epoch, logger):
        print("Training epoch {}".format(epoch))
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        end_time = time.time()

        progress_bar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc="Training",
            dynamic_ncols=True,
            colour="green",
        )
        for batch_idx, (inputs, labels) in progress_bar:
            data_time.update(time.time() - end_time)
            inputs = [input.to(self.device) for input in inputs]
            labels = labels.to(self.device)

            outputs = self.model(inputs[0], inputs[1], inputs[2])
            loss = self.criterion(outputs, labels)

            losses.update(loss.data, inputs[2].size(0))
            prec1, prec5 = calculate_accuracy(outputs.data, labels.data, topk=(1, 5))
            top1.update(prec1, inputs[2].size(0))
            top5.update(prec5, inputs[2].size(0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            del inputs, labels, outputs, loss

            progress_bar.set_postfix(
                {
                    "Loss": "{:.4f}".format(losses.avg),
                    "Acc@1": "{:.2f}".format(top1.avg),
                    "Acc@5": "{:.2f}".format(top5.avg),
                    "LR": self.optimizer.param_groups[0]["lr"],
                }
            )

        print(
            "Epoch {} Training - Average Loss: {:.4f}, Acc@1: {:.2f}, Acc@5: {:.2f}".format(
                epoch, losses.avg, top1.avg, top5.avg
            )
        )

        logger.log({
        'Epoch': epoch,
        'Train_Loss': losses.avg.item(),
        'Train_Acc@1': top1.avg.item(),
        'Train_Acc@5': top5.avg.item()
        })


    def validate(self, epoch, logger):
        print("Validation epoch {}".format(epoch))
        self.model.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        end_time = time.time()

        progress_bar = tqdm(
            enumerate(self.val_loader),
            total=len(self.val_loader),
            desc="Validation",
            dynamic_ncols=True,
            colour="blue",
        )
        for batch_idx, (inputs, labels) in progress_bar:
            data_time.update(time.time() - end_time)
            with torch.no_grad():
                inputs = [input.to(self.device) for input in inputs]
                labels = labels.to(self.device)
            outputs = self.model(inputs[0], inputs[1], inputs[2])
            loss = self.criterion(outputs, labels)
            prec1, prec5 = calculate_accuracy(outputs.data, labels.data, topk=(1, 5))
            top1.update(prec1, inputs[2].size(0))
            top5.update(prec5, inputs[2].size(0))

            losses.update(loss.data, inputs[2].size(0))

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            progress_bar.set_postfix(
                {
                    "Loss": "{:.4f}".format(losses.avg),
                    "Acc@1": "{:.2f}".format(top1.avg),
                    "Acc@5": "{:.2f}".format(top5.avg),
                }
            )

        print(
            "Epoch {} Validation - Average Loss: {:.4f}, Acc@1: {:.2f}, Acc@5: {:.2f}".format(
                epoch, losses.avg, top1.avg, top5.avg
            )
        )

        logger.log({
        'Epoch': epoch,
        'Val_Loss': losses.avg.item(),
        'Val_Acc@1': top1.avg.item(),
        'Val_Acc@5': top5.avg.item()
        })
        
    def save_checkpoint(self, epoch):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            f"checkpoints/model_epoch{epoch}.pth",
        )


if __name__ == "__main__":
    """
    python -m src.core.trainer
    """
    import sys
    import os

    project_root = os.path.abspath(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    )
    sys.path.append(project_root)
    from torch.utils.data import random_split
    from configs.hyperparameters import HyperParameters
    from src.data.dataset import get_training_set
    from src.data.dataloader import get_dataloader
    from src.transform.spatial_transforms import *
    from src.transform.temporal_transforms import *
    from src.transform.target_transforms import ClassLabel, VideoID
    from src.transform.target_transforms import Compose as TargetCompose
    from src.models import c2d

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

    model = c2d.get_model(num_classes=32)
    
    header = ['Epoch', 'Train_Loss', 'Train_Acc@1', 'Train_Acc@5', 'Val_Loss', 'Val_Acc@1', 'Val_Acc@5']
    logger = Logger(header)

    trainer = Trainer(opt, model, train_dataloader, val_dataloader)
    for epoch in range(opt.epochs):
        trainer.train_epoch(epoch, logger)
        trainer.validate(epoch, logger)
