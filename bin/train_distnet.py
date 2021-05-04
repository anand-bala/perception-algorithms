import math
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from perception_algos.datasets.kitti import (
    IMG_HEIGHT,
    IMG_SIZE,
    IMG_WIDTH,
    GetKITTIDistance,
)
from perception_algos.datasets.kitti import KITTIObjectDetect as KITTI
from perception_algos.models.distnet import BaseDistancePredictor as DistNet


class KITTIDistDataModule(pl.LightningDataModule):
    transform = transforms.Compose(
        [
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    target_transforms = GetKITTIDistance()

    def __init__(self, data_dir: Path, batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        KITTI(self.data_dir, train=True)

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            dataset = KITTI(
                self.data_dir,
                train=True,
                transform=self.transform,
                target_transform=self.target_transforms,
            )
            # Do a 60-40 split (approx)
            training_size = math.floor(0.6 * len(dataset))
            validation_size = len(dataset) - training_size
            self.training_data, self.validation_data = random_split(
                dataset, [training_size, validation_size]
            )

    def collate_batch(self, batch):
        """Given a map-style sample from the KITTI dataset, we return

        - The Tensor image (as it)
        - The Tensor boundi
        """
        zipped = tuple(zip(*batch))

        imgs, labels = zipped
        # Assume KITTI (we can figure other datasets out later...)
        # img will be a tuple of n tensors where n = batch_size
        # labels will be a tuple of list of dictionary labels.

        # We need to stack the images
        imgs = torch.stack(imgs)

        # From the labels, we need to create a list of n tensors (n = batch size), where
        # each tensor is of size [L, 4], where L is the number of bboxes in the
        # corresponding image.
        #
        # We also need to create a list of distance targets
        bboxes = []
        distances = []
        for label in labels:
            # We have a list of detections for each image
            bboxes.append(
                torch.stack(
                    [torch.as_tensor(det["bbox"], dtype=torch.float32) for det in label]
                )
            )
            distances.extend(
                torch.stack(
                    [
                        torch.as_tensor(det["distance"], dtype=torch.float32)
                        for det in label
                    ]
                )
            )
        d = torch.stack(distances).unsqueeze(-1)
        return imgs, bboxes, d

    def train_dataloader(self):
        return DataLoader(
            self.training_data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_batch,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation_data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_batch,
        )


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument(
        "--save-dir",
        type=lambda p: Path(p).expanduser().resolve(),
        default=Path.cwd(),
    )
    parser.add_argument("--dataset-dir", type=lambda p: Path(p).expanduser().resolve())
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--seed", default=1234, type=int)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(args.dataset_dir)
    batch_size = args.batch_size
    seed = args.seed
    pl.seed_everything(seed)

    # ------------
    # data
    # ------------
    data = KITTIDistDataModule(dataset_dir, batch_size)
    # ------------
    # model
    # ------------
    model = DistNet(IMG_SIZE)

    # ------------
    # training
    # ------------
    logger = pl_loggers.TensorBoardLogger(str(save_dir / "logs"))
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[
            EarlyStopping(monitor="val_loss"),
            ModelCheckpoint(dirpath=str(save_dir / "chk"), monitor="val_loss"),
        ],
        logger=logger,
    )
    trainer.fit(model, data)


if __name__ == "__main__":
    cli_main()
