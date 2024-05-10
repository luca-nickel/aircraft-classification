import os
from collections.abc import Callable
from typing import Optional, Tuple, Any

import PIL
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import verify_str_arg, download_and_extract_archive


class FGVCAircraft_bbox(VisionDataset):
    def __init__(
            self,
            root: str,
            split: str = "trainval",
            annotation_level: str = "variant",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._split = verify_str_arg(split, "split", ("train", "val", "test"))
        self._annotation_level = verify_str_arg(
            annotation_level, "annotation_level", "bounding_box"
        )

        self._data_path = os.path.join(self.root, "fgvc-aircraft-2013b")
        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        annotation_file = os.path.join(
            self._data_path,
            "data",
            {
                "bounding_box": "images_bounding_box.txt"
            }[self._annotation_level],
        )
        image_data_folder = os.path.join(self._data_path, "data", "images")
        self._image_files = []
        self._labels = []
        with open(annotation_file, "r") as f:
            self.classes = []
            if annotation_level == "bounding_box":
                for line in f:
                    parts = line.split()
                    # Extract the last four figures
                    image_name = parts[0]
                    self._image_files.append(os.path.join(image_data_folder, f"{image_name}.jpg"))
                    coordinates_label = parts[-4:]
                    self._labels.append(coordinates_label)

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def _download(self) -> None:
        """
        Download the FGVC Aircraft dataset archive and extract it under root.
        """
        if self._check_exists():
            return
        download_and_extract_archive(self._URL, self.root)

    def _check_exists(self) -> bool:
        return os.path.exists(self._data_path) and os.path.isdir(self._data_path)
