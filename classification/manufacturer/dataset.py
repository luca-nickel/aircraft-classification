from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from torchvision.datasets.folder import default_loader
import torchvision.transforms as transforms

class CustomDataset(Dataset):
    def __init__(self, file_path, data_root, classes, transform=None):
        self.data_root = data_root
        self.transform = transform
        self.classes = classes
        self.samples = []

        with open(file_path, 'r') as f:
            for line in f:
                image_name, manufacturer = line.strip().split(" ", 1)
                image_path = data_root / "data" / "images" / f"{image_name}.jpg"
                if not image_path.exists():
                    continue  # Skip missing files
                label = classes.index(manufacturer)
                self.samples.append((image_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = default_loader(str(image_path))
        if self.transform:
            image = self.transform(image)
        return image, label

def load_classes(file_path):
    with open(file_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

def get_dataloaders(data_root, batch_size=4, num_workers=2):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    manufacturers_file = data_root / "data" / "manufacturers.txt"
    classes = load_classes(manufacturers_file)

    train_file = data_root / "data" / "images_manufacturer_train.txt"
    test_file = data_root / "data" / "images_manufacturer_test.txt"

    trainset = CustomDataset(train_file, data_root, classes, transform)
    testset = CustomDataset(test_file, data_root, classes, transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader, len(classes)
