import yaml
from typing import Dict, Any


class HyperParameters:
    def __init__(self, path: str):
        self.config = self._load_config(path)
        self._init_parameters()

    def _load_config(self, path: str) -> Dict[str, Any]:
        try:
            with open(path) as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file {path} does not exist")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Configuration file {path} is incorrectly formatted: {e}")

    def _init_parameters(self):
        self.no_mean_norm = self._get_param("normalization", "no_mean_norm", False)
        self.std_norm = self._get_param("normalization", "std_norm", True)
        self.mean = self._get_param("normalization", "mean", [0.485, 0.456, 0.406])
        self.std = self._get_param("normalization", "std", [0.229, 0.224, 0.225])

        self.video_path = self._get_param("training", "video_path", "dataset/")
        self.annotation_path = self._get_param("training", "annotation_path", "annotation/")
        self.batch_size = self._get_param("training", "batch_size", 16)
        self.num_workers = self._get_param("training", "num_workers", 4)
        self.modality = self._get_param("training", "modality", "RGB")
        self.no_train = self._get_param("training", "no_train", False)
        self.train_crop = self._get_param("training", "train_crop", "random")
        self.scales = self._get_param("training", "scales", [1.0, 0.8, 0.6])
        self.sample_size = self._get_param("training", "sample_size", 224)
        self.sample_duration = self._get_param("training", "sample_duration", 16)
        self.sensor_duration = self._get_param("training", "sensor_duration", 10)
        self.downsample = self._get_param("training", "downsample", 2)
        self.norm_value = self._get_param("training", "norm_value", 255)
        self.lr = self._get_param("training", "lr", 0.01)
        self.epochs = self._get_param("training", "epochs", 100)
        self.optimizer_type = self._get_param("training", "optimizer_type", 'adam')
        self.num_classes = self._get_param("training", "num_classes", 32)

    def _get_param(self, section: str, key: str, default: Any) -> Any:
        return self.config.get(section, {}).get(key, default)
