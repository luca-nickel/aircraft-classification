import os
from collections.abc import Callable
from typing import Optional, Tuple, Any

import PIL
import torch
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive


class FgvcAircraftBbox(VisionDataset):
    def __init__(
        self,
        root: str,
        file: str = "images_bounding_box_train.txt",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.file = file
        self._data_path = os.path.join(self.root, "fgvc-aircraft-2013b")
        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        image_data_folder = os.path.join(self._data_path, "data", "images")
        annotation_file = os.path.join(self._data_path, "data", self.file)
        self._image_files = []
        self._labels = []
        with open(annotation_file, "r") as f:
            for line in f:
                parts = line.split()
                # Extract the last four figures
                image_name = parts[0]
                self._image_files.append(
                    os.path.join(image_data_folder, f"{image_name}.jpg")
                )
                coordinates_str_label = parts[-4:]
                coordinates_label = [float(i) for i in coordinates_str_label]
                self._labels.append(torch.FloatTensor(coordinates_label))

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            print("TEST")
            print(image.size)
            image = self.transform(image, 1600)

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
