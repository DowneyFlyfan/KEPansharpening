from dataclasses import dataclass, field
from itertools import accumulate
import torch
import time
import os


@dataclass
class BaseConfig:
    _dtype: torch.dtype = field(
        default=torch.float32, metadata={"help": "data type"}
    )  # WARN: Dtype Changed
    data: str = field(
        default="wv3_reduced",
        metadata={
            "help": "Dataset Selection",
            "choices": [
                "wv3_reduced",
                "wv3_reduced",
                "wv2_origscale",
                "wv3_origscale",
                "qb_reduced",
                "qb_origscale",
                "gf2_reduced",
                "gf2_origscale",
            ],
        },
    )
    ratio: int = 4

    def __post_init__(self):
        self.device_setup()
        self.validate_data()
        self.setup_data_config()
        self.aug_size = int(self.test_sidelen * 0.5)

    def device_setup(self):
        """Check Available Devices"""
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

    def validate_data(self):
        valid_choices = self.__dataclass_fields__["data"].metadata["choices"]
        if self.data not in valid_choices:
            raise ValueError(f"无效数据集: {self.data}，可选值: {valid_choices}")

    def setup_data_config(self):
        config_map = {
            "wv2_origscale": (
                "./test_data/test_wv2_OrigScale.h5",
                512,
                "WV2",
                8,
                2047.0,
            ),
            "wv2_reduced": ("./test_data/test_wv2_Reduced.h5", 256, "WV2", 8, 2047.0),
            "wv3_origscale": (
                "./test_data/test_wv3_OrigScale.h5",
                512,
                "WV3",
                8,
                2047.0,
            ),
            "wv3_reduced": ("./test_data/test_wv3_Reduced.h5", 256, "WV3", 8, 2047.0),
            "qb_reduced": ("./test_data/test_qb_Reduced.h5", 256, "QB", 4, 2047.0),
            "qb_origscale": ("./test_data/test_qb_OrigScale.h5", 512, "QB", 4, 2047.0),
            "gf2_reduced": ("./test_data/test_gf2_Reduced.h5", 256, "QB", 4, 2047.0),
            "gf2_origscale": (
                "./test_data/test_gf2_OrigScale.h5",
                512,
                "QB",
                4,
                2047.0,
            ),
        }
        self.test_path, self.test_sidelen, self.sensor, self.channel, self.max_value = (
            config_map[self.data]
        )


@dataclass
class TrainArgs(BaseConfig):
    lr: dict = field(
        default_factory=lambda: {
            "backbone": 4e-3,
            "pan_pred_net": 2.5e-3,
            "mtfnet": 1e-3,
            "mranet": 2e-3,
        },
        metadata={"help": "learning rate of different components"},
    )

    min_lr: dict = field(
        default_factory=lambda: {
            "backbone": 5e-4,
            "pan_pred_net": 1e-3,
            "mranet": 1e-3,
        },
        metadata={"help": "min learning rate in scheduler"},
    )

    scheduler_type: dict = field(
        default_factory=lambda: {
            "backbone": "flat_down",
            "pan_pred_net": "flat_down",
            "mranet": "flat_down",
        },
        metadata={"help": "learning rate of different components"},
    )

    epoch: dict = field(
        default_factory=lambda: (lambda d: {**d, "csum": list(accumulate(d.values()))})(
            {
                "stageI": 500,
                "stageII": 100,
                "stageIII": 800,
                # "stageIV": 100,
                # "stageV": 200,
            }
        ),
        metadata={"help": "epoch settings"},
    )

    update_steps: int = 5
    cosine_point: float = 0.7
    flat_point: float = 0.2

    mixed: bool = True  # Whether or not using mixed precision

    # loss
    lambda_mssam: float = 2
    lambda_msmse: float = 3
    lambda_panmse: float = 4

    def __post_init__(self):
        super().__post_init__()
        self.epoch["check"] = 100
        self.epoch["save"] = 10

        self.epoch["cross"] = 100
        self.epoch["pretrain"] = 100

        self.epoch["total"] = self.epoch["csum"][-1]
        exp_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.result_path = "test_results/Exp" + exp_time


@dataclass
class ModelArgs(BaseConfig):
    pad_type: str = "reflection"
    hidden_dim: int = 64

    def __post_init__(self):
        super().__post_init__()


@dataclass
class DataArgs(BaseConfig):
    validate_batch_size: int = 54
    train_batch_size: int = 54
    validate_path: str = "./training_data/valid_wv3.h5"
    train_path: str = "./training_data/train_wv3.h5"
    max_value: float = 2047.0

    def __post_init__(self):
        super().__post_init__()
        self.workers = os.cpu_count() - 2


bargs = BaseConfig()
targs = TrainArgs()
dargs = DataArgs()
margs = ModelArgs()
