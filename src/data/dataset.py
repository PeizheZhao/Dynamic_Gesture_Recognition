import torch
from torch.utils.data import Dataset
import json
import os
import functools
from PIL import Image
import pandas as pd
import math
from typing import *


def pil_loader(path, modality):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        # print(path)
        with Image.open(f) as img:
            if modality == "RGB":
                return img.convert("RGB")
            elif modality == "Flow":
                return img.convert("L")
            elif modality == "Depth":
                return img.convert(
                    "L"
                )  # 8-bit pixels, black and white check from https://pillow.readthedocs.io/en/3.0.x/handbook/concepts.html


def accimage_loader(path, modality):
    try:
        import accimage

        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


# try to use accimage first, if fail use pil
def get_default_image_loader():
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader
    else:
        return pil_loader


def video_loader(
    video_dir_path, frame_indices, modality, sample_duration, image_loader
) -> Dict[str, List[Any]]:
    video = {"up": [], "down": []}
    if modality == "RGB":
        for i in frame_indices:
            # edit
            image_path = os.path.join(video_dir_path, "up_hand", f"{i}.jpg")
            if os.path.exists(image_path):
                video["up"].append(image_loader(image_path, modality))
            else:
                print(image_path, "------- Does not exist")
                return video
            image_path = os.path.join(video_dir_path, "down_hand", f"{i}.jpg")
            if os.path.exists(image_path):
                video["down"].append(image_loader(image_path, modality))
            else:
                print(image_path, "------- Does not exist")
                return video
    elif modality == "Depth":
        for i in frame_indices:
            image_path = os.path.join(
                video_dir_path.rsplit(os.sep, 2)[0],
                "Depth",
                "depth" + video_dir_path[-1],
                "{:06d}.jpg".format(i),
            )
            if os.path.exists(image_path):
                video.append(image_loader(image_path, modality))
            else:
                print(image_path, "------- Does not exist")
                return video
    elif modality == "RGB-D":
        for i in frame_indices:  # index 35 is used to change img to flow
            image_path = os.path.join(video_dir_path, "{:06d}.jpg".format(i))
            image_path_depth = os.path.join(
                video_dir_path.rsplit(os.sep, 2)[0],
                "Depth",
                "depth" + video_dir_path[-1],
                "{:06d}.jpg".format(i),
            )
            image = image_loader(image_path, "RGB")
            image_depth = image_loader(image_path_depth, "Depth")
            if os.path.exists(image_path):
                video.append(image)
                video.append(image_depth)
            else:
                print(image_path, "------- Does not exist")
                return video
    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def load_annotation_data(data_file_path):
    with open(data_file_path, "r") as data_file:
        return json.load(data_file)


def get_video_names_and_annotations(
    data, subset
) -> Tuple[List[str], List[Dict[str, int]]]:
    video_names = []
    annotations = []

    for key, value in data["database"].items():
        this_subset = value["subset"]
        if this_subset in subset:
            label = value["annotations"]["label"]
            video_names.append(key.split("_")[0])
            annotations.append(value["annotations"])

    return video_names, annotations


def get_class_labels(data):
    label_classes_map = data["label_map"]
    return label_classes_map


def get_sensor(video_dir_path, frame_indices, sensor_duration, total_frames):
    csv_path = os.path.join(video_dir_path, "sensor", "sensor.csv")
    df = pd.read_csv(csv_path)
    total_rows = len(df)
    total_sensor = sensor_duration * total_frames
    assert (
        total_rows > total_sensor
    ), f"[Error]: Sensor duration do not support dataset \n Need: {total_sensor} But: {total_rows}"

    def uniform_sampling(df, total_sensor):
        data = df
        target_rows = total_sensor
        step = len(data) / target_rows
        compressed_data = data.iloc[[int(i * step) for i in range(target_rows)]]
        return compressed_data

    def moving_window_average(df, total_sensor):
        data = df
        target_rows = total_sensor
        window_size = int(len(data) / target_rows)
        compressed_data = data.groupby(data.index // window_size).mean()
        return compressed_data

    df = uniform_sampling(df, total_sensor)
    sensor_data = pd.DataFrame()
    for i in frame_indices:
        begin = i * sensor_duration
        sensor_data = pd.concat(
            [sensor_data, df.iloc[begin : begin + sensor_duration, 1:]], axis=0
        )
    return torch.tensor(sensor_data.values, dtype=torch.float32)


def make_dataset(
    root_path, annotation_path, subset, n_samples_for_each_video, sample_duration
) -> Tuple[List[Dict[str, any]], Dict[int, str]]:
    if type(subset) == list:
        subset = subset
    else:
        subset = [subset]
    data = load_annotation_data(annotation_path)
    video_names, annotations = get_video_names_and_annotations(data, subset)
    idx_to_class = get_class_labels(data)
    class_to_idx = {}
    for name, label in idx_to_class.items():
        class_to_idx[label] = name
    dataset = []
    list_subset = ""
    for x in subset:
        list_subset += x + ","
    print("[INFO]: EgoGesture Dataset - " + list_subset + " is loading...")
    for i in range(len(video_names)):
        # if i % 100 == 0:
        # print("dataset loading [{}/{}]".format(i, len(video_names)))

        video_path = os.path.join(root_path, video_names[i])

        if not os.path.exists(video_path):
            print(video_path + " does not exist")
            continue

        jpg_path = os.path.join(video_path, "up_hand")
        jpg_count = len([f for f in os.listdir(jpg_path) if f.endswith(".jpg")])

        begin_t = int(float(annotations[i]["start_frame"]))
        end_t = int(float(annotations[i]["end_frame"]))
        n_frames = end_t - begin_t + 1
        sample = {
            "video": video_path,
            "segment": [begin_t, end_t],
            "n_frames": n_frames,
            "video_id": i,
            "total_frames": jpg_count,
        }
        if len(annotations) != 0:
            sample["label"] = annotations[i]["label"]
        else:
            sample["label"] = -1

        if n_samples_for_each_video == 1:
            sample["frame_indices"] = list(range(begin_t, end_t + 1))
            dataset.append(sample)
        """else:
            if n_samples_for_each_video > 1:
                step = max(1,
                           math.ceil((n_frames - 1 - sample_duration) /
                                     (n_samples_for_each_video - 1)))
            else:
                step = sample_duration
            for j in range(1, n_frames, step):
                sample_j = copy.deepcopy(sample)
                sample_j['frame_indices'] = list(
                    range(j, min(n_frames + 1, j + sample_duration)))
                dataset.append(sample_j)"""

    return dataset, idx_to_class


class ClassificationDataset(Dataset):
    def __init__(
        self,
        root_path,
        annotation_path,
        subset,
        n_samples_for_each_video=1,
        spatial_transform=None,
        temporal_transform=None,
        target_transform=None,
        sample_duration=16,
        modality="RGB",
        get_loader=get_default_video_loader,
        sensor_duration=10,
    ):

        if subset == "training":
            subset = ["training", "validation"]
        self.data, self.class_names = make_dataset(
            root_path,
            annotation_path,
            subset,
            n_samples_for_each_video,
            sample_duration,
        )

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.modality = modality
        self.sample_duration = sample_duration
        self.sensor_duration = sensor_duration
        self.loader = get_loader()
        self.sensor = get_sensor

    def __getitem__(self, index):
        path = self.data[index]["video"]

        frame_indices = self.data[index]["frame_indices"]

        total_frames = self.data[index]["total_frames"]

        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)

        clips = self.loader(path, frame_indices, self.modality, self.sample_duration)
        for key, clip in clips.items():
            if self.spatial_transform is not None:
                self.spatial_transform.randomize_parameters()
                clip = [self.spatial_transform(img) for img in clip]

            im_dim = clip[0].size()[-2:]
            clip = (
                torch.cat(clip, 0)
                .view((self.sample_duration, -1) + im_dim)
                .permute(1, 0, 2, 3)
            )
            clips[key] = clip

        sensor = self.sensor(path, frame_indices, self.sensor_duration, total_frames)

        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return clips["up"], clips["down"], sensor, target

    def __len__(self):
        return len(self.data)


def get_training_set(opt, spatial_transform, temporal_transform, target_transform):
    if opt.no_train:
        subset = ["training", "validation"]
    else:
        subset = "training"
    training_data = ClassificationDataset(
        opt.video_path,
        opt.annotation_path,
        subset,
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        target_transform=target_transform,
        sample_duration=opt.sample_duration,
        sensor_duration=opt.sensor_duration,
        modality=opt.modality,
    )
    return training_data
