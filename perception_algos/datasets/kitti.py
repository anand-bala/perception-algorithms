import csv
import os
from pathlib import Path
from typing import Any, Callable, List, Mapping, NamedTuple, Optional, Tuple, Union

import numpy as np
import numpy.linalg as la
import pytorch_lightning as pl
from PIL import Image
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets.vision import VisionDataset

PathLike = Union[str, "os.PathLike[Any]"]

IMG_WIDTH = 32 * (1224 // 32)
IMG_HEIGHT = 32 * (370 // 32)
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
NUM_CLASSES = 8
LABEL_MAP = {
    "Car": 0,
    "Van": 1,
    "Truck": 2,
    "Pedestrian": 3,
    "Person_sitting": 4,
    "Cyclist": 5,
    "Tram": 6,
    "Misc": 7,
}


class KITTIObjectDetect(VisionDataset):
    """Dataset loader for the KITTI object detection dataset.

    Parameters
    ==========

    data_dir: PathLike
        Path to KITTI dataset root (e.g. kitti/object_detection). Should contain a
        `training` and `testing` directory, each with a `image_2` directory containing
        PNG images. The `training/label_2` directory should contain a bunch of TXT
        labels.

        Expects the following folder structure if download=False:

        .. code::

            <root>
                ├── training
                |   ├── image_2
                |   └── label_2
                └── testing
                    └── image_2

    train: Optional[bool]
        Use `train` split if true, else `test` split. Defaults to `train`.
    transform: Optional[Callable]
        A function/transform that takes in a PIL image and returns a transformed
        version. E.g, ``transforms.ToTensor``
    target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    transforms: Optional[Callable]
        A function/transform that takes input sample and its target as entry and returns
        a transformed version.
    download : Optional[bool]
        If true, downloads the dataset from the internet and puts it in root directory.
        If dataset is already downloaded, it is not downloaded again.
    """

    data_url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/"
    resources = [
        "data_object_image_2.zip",
        "data_object_label_2.zip",
    ]
    image_dir_name = "image_2"
    labels_dir_name = "label_2"

    def __init__(
        self,
        data_dir: PathLike,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        download: bool = False,
    ):
        super().__init__(
            str(data_dir),
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )
        self.dataset_path = Path(data_dir).expanduser().resolve()
        assert self.dataset_path.is_dir(), "Given data path is not a directory"
        self.train = train
        self._location = "training" if self.train else "testing"

        if download:
            self.download()

        if self.train:
            assert (
                self.dataset_path / "training" / "image_2"
            ).is_dir(), (
                "Given data path doesn't contain a subdirectory `training/image_2`"
            )
            assert (
                self.dataset_path / "training" / "label_2"
            ).is_dir(), (
                "Given data path doesn't contain a subdirectory `training/label_2`"
            )
        else:
            assert (
                self.dataset_path / "testing" / "image_2"
            ).is_dir(), (
                "Given data path doesn't contain a subdirectory `testing/image_2`"
            )

        self._data = [
            img.stem
            for img in sorted(
                (self.dataset_path / self._location / "image_2").glob("*.png")
            )
        ]

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> Tuple[Any, Optional[List[Mapping]]]:
        """Get image and label at index.

        Parameters
        ==========

        index: int
            Index in the dataset

        Returns
        =======

        This returns multiple values in the following order:

        image: Any
            Image from the dataset the the index read with shape `(n_channels,
            image_height, image_width)`, and passed through `transforms`.
        labels: Optional[List[Mapping]]
            If `train` is `True`, return a list of dictionaries with the following keys:

            - type: str
            - truncated: float
            - occluded: int
            - alpha: float
            - bbox: float[4]
            - dimensions: float[3]
            - locations: float[3]
            - rotation_y: float

            Otherwise, the output is `None`. The labels can be transformed through `target_transform`.

        transform : Optional[List[Callable]]
            A function/transform that takes in a PIL image and returns a transformed
            version. E.g, `transforms.ToTensor`
        """
        stem = self._data[index]
        img_file = self.dataset_path / self._location / "image_2" / (stem + ".png")
        image = Image.open(img_file.resolve())  # type: Image
        labels = self._parse_label(stem) if self.train else None
        if self.transforms:
            image, labels = self.transforms(image, labels)
        return image, labels

    def _parse_label(self, stem: str) -> List[Mapping]:
        target = []
        label = self.dataset_path / self._location / "label_2" / (stem + ".txt")
        with open(label, "r") as inp:
            content = csv.reader(inp, delimiter=" ")
            for line in content:
                t = {
                    "type": line[0],
                    "truncated": float(line[1]),
                    "occluded": int(line[2]),
                    "alpha": float(line[3]),
                    "bbox": [float(x) for x in line[4:8]],
                    "dimensions": [float(x) for x in line[8:11]],
                    "location": [float(x) for x in line[11:14]],
                    "rotation_y": float(line[14]),
                }
                if t["type"] == "DontCare":
                    continue
                else:
                    target.append(t)

        return target

    def _check_exists(self) -> bool:
        ok = True
        if self.train:
            ok = ok and (self.dataset_path / self._location / "label_2").is_dir()
        ok = ok and (self.dataset_path / self._location / "image_2").is_dir()
        return ok

    def download(self) -> None:
        """Download the KITTI data if it doesn't exist already."""

        if self._check_exists():
            return

        os.makedirs(self.dataset_path, exist_ok=True)

        # download files

        for fname in self.resources:
            download_and_extract_archive(
                url=f"{self.data_url}{fname}",
                download_root=str(self.dataset_path),
                filename=fname,
            )


class GetKITTIDistance(object):
    """
    Get the distance to an object using the KITTI object detection labels.

    .. note::

        This is an approximation of the distance using the location of the bbox. There
        are more sophisticated ways of doing this, e.g., by using the Velodyne point
        cloud, but it needs more information.
    """

    def __call__(self, targets: List[Mapping]) -> List[Mapping]:
        assert all(("location" in target.keys()) for target in targets)
        assert all(len(target["location"]) == 3 for target in targets)

        new_targets = []

        for t in targets:
            new_t = dict(**t)

            loc = np.asarray(t["location"])
            dist = la.norm(loc, 2)

            new_t["distance"] = dist
            new_targets.append(new_t)

        return new_targets


class StrLabelsToInt(object):
    """Convert string labels into integer labels"""

    def __call__(self, targets: List[Mapping]) -> List[Mapping]:
        assert all(("type" in target.keys()) for target in targets)
        assert all(isinstance(target["type"], str) for target in targets)

        new_targets = []
        for t in targets:
            new_t = dict(**t)

            new_t["label"] = LABEL_MAP[t["type"]]
            new_targets.append(new_t)

        return new_targets
