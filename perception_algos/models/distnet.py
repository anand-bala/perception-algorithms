from collections import OrderedDict
from functools import reduce
from typing import List, Tuple, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics.functional import mean_squared_error
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torchvision.models.resnet import resnet50
from torchvision.models.vgg import vgg16
from torchvision.ops.roi_pool import RoIPool


class BaseDistancePredictor(pl.LightningModule):
    """Base distance predictor neural network described in [zhu_learning_2019]_.

    .. [zhu_learning_2019] J. Zhu and Y. Fang, "Learning Object-Specific Distance From a
        Monocular Image," 2019, pp. 3839–3848.

    The model takes as input an image an a bounding box, and outputs a distance
    prediction for that image and bounding box.

    The input image needs to be a tensor with shape `(n_channels,  image_height,
    image_width)`

    Parameters
    ----------

    image_size : (int, int)
        The height and width of the image
    feature_extractor : str
        One of "vgg16" or "resnet50"
    """

    def __init__(
        self, image_size: Tuple[int, int], feature_extractor: str = "resnet50"
    ):
        super().__init__()
        self._image_size = image_size
        if feature_extractor not in ["resnet50", "vgg16"]:
            raise ValueError("feature_extractor must be one of 'resnet50' or 'vgg16'")

        self._feature_extractor = feature_extractor

        self.learning_rate = 0.01

        self.roi_pool = RoIPool(
            output_size=(7, 7),
            spatial_scale=1 / 32,
        )

        if self._feature_extractor == "resnet50":
            # Remove the pooling and fc layers of resnet50
            resnet_layers = list(resnet50(pretrained=True).named_children())
            resnet_feature_layers = resnet_layers[:-2]
            max_pool_layer = nn.AdaptiveMaxPool2d(
                output_size=tuple(
                    map(lambda c: c // 32, self._image_size)
                ),  # Scaled down size of 32
            )
            self.features = nn.Sequential(
                OrderedDict([*resnet_feature_layers, ("max_pool", max_pool_layer)])
            )
            features_n_channels = 2048
            self.fc = nn.Sequential(  # Input is the flattened roi pool
                nn.Linear(in_features=(features_n_channels * 7 * 7), out_features=1024),
                nn.Linear(in_features=1024, out_features=512),
                nn.Linear(in_features=512, out_features=1),
                nn.Softplus(),
            )

        elif self._feature_extractor == "vgg16":
            self.features = vgg16(pretrained=True).features  # Already scaled down 32
            features_n_channels = 512
            self.fc = nn.Sequential(  # Input is the flattened roi pool
                nn.Linear(in_features=(features_n_channels * 7 * 7), out_features=2048),
                nn.Linear(in_features=1024, out_features=512),
                nn.Linear(in_features=512, out_features=1),
                nn.Softplus(),
            )

        else:
            raise ValueError(f"Unsupported backbone: {self._feature_extractor}")

        # TODO: Freeze the backbone?
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(
        self, img: torch.Tensor, bboxes: Union[torch.Tensor, List[torch.Tensor]]
    ) -> torch.Tensor:
        # First, we pass img through the feature extractor
        features = self.features(img)
        roi_pooled = self.roi_pool(features, bboxes)
        vector = torch.flatten(roi_pooled, 1, -1)
        dist = self.fc(vector)
        return dist

    def training_step(self, batch, batch_idx):
        imgs, bboxes, d = batch

        d_hat = self(imgs, bboxes)

        loss = F.smooth_l1_loss(d_hat, d)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, bboxes, d = batch

        d_hat = self(imgs, bboxes)

        loss = F.smooth_l1_loss(d_hat, d)
        mse = mean_squared_error(d_hat, d)
        self.log_dict({"val_loss": loss, "mse": mse}, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
