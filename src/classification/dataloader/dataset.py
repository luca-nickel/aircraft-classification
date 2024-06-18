import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
from preprocessing import transforms_service


class ClassificationDataset(data.Dataset):
    def __init__(
        self,
        seed=42,
        file="../data/input/fgvc-aircraft-2013b/data/images_manufacturer_train.txt",
        transforms=None,
        train_transforms=None,
        classes=None,
    ):
        self.picture_path = file.replace(file.split("/")[-1], "images")
        np.random.seed(seed)
        self.transforms = transforms
        self.train_transforms = train_transforms

        with open(file, "r") as f:
            self.data = f.readlines()
        self.data = np.array(
            [
                (
                    line.strip().split(" ")[0],
                    " ".join(line.strip().split(" ")[1:]),
                    0,
                )
                for line in self.data
            ]
        )

        self.data[:, 1] = np.where(
            (self.data[:, 1] != "Airbus") & (self.data[:, 1] != "Boeing"),
            "Other",
            self.data[:, 1],
        )

        if classes:
            print("Classes are provided")
            self.classes = classes
            self.data = self.data[np.isin(self.data[:, 1], self.classes)]
        else:
            self.classes = sorted(list(set(self.data[:, 1])))

        if self.train_transforms is not None:
            print("balancing classes!")
            # Count the number of occurrences of each class
            unique, counts = np.unique(self.data[:, 1], return_counts=True)
            max_count = np.max(counts)
            # Balance the classes
            for cls in self.classes:
                cls_indices = np.where(self.data[:, 1] == cls)[0]
                cls_count = len(cls_indices)
                if cls_count < max_count:
                    # Repeat the entries of the under-represented class
                    extra_indices = np.repeat(cls_indices, (max_count // cls_count) - 1)
                    # Randomly select the remaining entries to match the max_count exactly
                    if len(extra_indices) < max_count - cls_count:
                        extra_indices = np.concatenate(
                            (
                                extra_indices,
                                np.random.choice(
                                    cls_indices,
                                    max_count - cls_count - len(extra_indices),
                                    replace=False,
                                ),
                            )
                        )
                    # Add the extra entries to the data
                    self.data = np.concatenate((self.data, self.data[extra_indices]))
                    # Set the transform flag for the extra entries
                    self.data[cls_indices, 2] = 1

        print(self.data.shape)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        # Transforms can now be used directly on tensors
        img = Image.open(f"{self.picture_path}/{self.data[index, 0]}.jpg").convert(
            "RGB"
        )
        label = torch.tensor(self.classes.index(self.data[index, 1]))
        if self.transforms is not None:
            img = self.transforms(img)
        if self.data[index, 2] == 1:
            img = self.train_transforms(img)

        return img, label


def main():
    dataset = ClassificationDataset()
    # print(dataset)
    print(dataset[0])


if __name__ == "__main__":
    main()
