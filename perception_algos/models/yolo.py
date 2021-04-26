import enum

import torch


@enum.unique
class Models(enum.Enum):
    V5_SM = "yolov5s"
    V5_MD = "yolov5m"
    V5_LG = "yolov5l"
    V5_XL = "yolov5x"


def get_model(arch: Models, freeze=True) -> torch.nn.Module:
    """Get a YOLOv5 movel with the given architecture.

    Parameters
    ==========

    arch:   Models
            The YOLOv5 architecture
    freeze: bool
            Freeze the model?
    """
    model = torch.hub.load("ultralytics/yolov5", arch.name)  # type: torch.nn.Module
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    return model
