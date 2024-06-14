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
    ):
        self.picture_path = file.replace(file.split("/")[-1], "images")
        np.random.seed(seed)
        self.transforms = transforms

        with open(file, "r") as f:
            self.data = f.readlines()
        self.data = [x.strip().split(" ") for x in self.data]
        self.classes = sorted(list(set([x[1] for x in self.data])))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Transforms can now be used directly on tensors
        img = Image.open(f"{self.picture_path}/{self.data[index][0]}.jpg").convert(
            "RGB"
        )
        label = torch.tensor(self.classes.index(self.data[index][1]))
        if self.transforms is not None:
            img = self.transforms(img)

        return img, label


def main():
    dataset = ClassificationDataset()
    # print(dataset)
    print(dataset[0])


if __name__ == "__main__":
    main()
