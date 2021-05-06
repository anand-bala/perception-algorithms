import enum
import os
from typing import Any, Union

import pytorch_lightning as pl
import torch

PathLike = Union[str, "os.PathLike[Any]"]


@enum.unique
class Models(enum.Enum):
    V5_SM = "yolov5s"
    V5_MD = "yolov5m"
    V5_LG = "yolov5l"
    V5_XL = "yolov5x"


def get_model(arch: Models) -> torch.nn.Module:
    """Get a YOLOv5 movel with the given architecture.

    Parameters
    ==========

    arch:   Models
            The YOLOv5 architecture
    """
    model = torch.hub.load("ultralytics/yolov5", arch.value)  # type: torch.nn.Module
    return model


def load_model(weights_path: str) -> torch.nn.Module:
    """Load a YOLOv5 model with given weights

    Parameters
    ----------

    weights_path: PathLike
                  Path to the weights to load
    """
    model = torch.hub.load("ultralytics/yolov5", "custom", path=str(weights_path))
    return model


class YOLO(pl.LightningModule):
    """Wrapper around ultralytics/yolov5"""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.detector = model

    def forward(self, img: torch.Tensor):
        return self.detector(img)
