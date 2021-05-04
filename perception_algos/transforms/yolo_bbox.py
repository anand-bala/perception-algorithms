from typing import List, Mapping, Tuple


class ToYOLOLabels(object):
    """
    Provides a transform to convert bounding boxes in different datasets into YOLO
    compatible bounding boxes with the format:

        <object id> <x> <y> <width> <height>

    Parameters
    ----------

    from : str
        Dataset to convert from. Currently supports:

        - "kitti": The KITTI object detection dataset.
    class_map : Mapping[str, int]
        A Mapping from string classes to integer class IDs
    """

    def __init__(self, _from: str, class_map: Mapping[str, int]):
        if _from not in ["kitti"]:
            raise ValueError("Unsupported dataset to convert from: {}".format(_from))
        self._from = "kitti"
        self._class_map = class_map

    def __call__(self, target: List[Mapping]) -> List[Mapping]:
        if self._from == "kitti":
            return self._from_kitti(target)
        return target

    def _from_kitti(self, target: List[Mapping]) -> List[Mapping]:
        assert all(
            ("type" in t.keys()) for t in target
        ), "Given target labels doesn't match KITTI labels"
        assert all(
            ("bbox" in t.keys()) for t in target
        ), "Given target labels doesn't match KITTI labels"
        assert all(
            len(t["bbox"]) == 4 for t in target
        ), "Bounding Box needs to have 4 values (x1, y1, x2, y2)"

        KNOWN_LABELS = set(
            {
                "Car",
                "Van",
                "Truck",
                "Pedestrian",
                "Person_sitting",
                "Cyclist",
                "Tram",
                "Misc",
            }
        )
        assert KNOWN_LABELS == self._class_map.keys()

        IMG_WIDTH = 1224
        IMG_HEIGHT = 370

        new_target = []

        for t in target:
            x1, y1, x2, y2 = map(float, t["bbox"])
            # Normalize the bbox
            x1 = x1 / IMG_WIDTH
            x2 = x2 / IMG_WIDTH
            y1 = y1 / IMG_HEIGHT
            y2 = y2 / IMG_HEIGHT

            # Compute centers and width
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            width = abs(x2 - x1)
            height = abs(y2 - y1)

            bbox = [cx, cy, width, height]
            id = self._class_map[t["type"]]

            new_t = dict(**t)  # Copy over the t

            new_t["bbox"] = bbox
            new_t["id"] = id
            new_target.append(new_t)

        return new_target


class FromYOLOLabels(object):
    """
    Provides a transform to convert bounding boxes from YOLO format to other formats

    Parameters
    ----------

    from : str
        Dataset to convert from. Currently supports:

        - "kitti": The KITTI object detection dataset.
    img_size : Tuple[int, int]
        Height and Width of the images.
    """

    def __init__(self, _from: str, img_size: Tuple[int, int]):
        if _from not in ["kitti"]:
            raise ValueError("Unsupported dataset to convert from: {}".format(_from))
        self._from = "kitti"
        self._img_size = img_size

    def __call__(self, target: Mapping) -> Mapping:
        new_target = dict(**target)
        img_height, img_width = self._img_size

        assert "bbox" in target.keys()
        assert len(target["bbox"]) == 4
        assert all(
            map(lambda c: 0 < c <= 1, target["bbox"])
        ), "bbox is not normalized (YOLO format)"

        cx, cy, w, h = target["bbox"]

        left = (cx - (w / 2)) * img_width
        right = (cx + (w / 2)) * img_width
        top = (cy - (h / 2)) * img_height
        bottom = (cy + (h / 2)) * img_height

        bbox = [left, top, right, bottom]
        new_target["bbox"] = bbox
        return new_target
