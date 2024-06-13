import torch
from pathlib import Path
from torchvision.datasets.folder import default_loader
import torchvision.transforms as transforms

def load_classes(file_path):
    with open(file_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

def load_dataset(file_path, data_root, classes, transform=None):
    dataset = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            print(line)
            line = line.strip()
            print(line)
            parts = line.split(" ", 1)  # Nach dem ersten Leerzeichen aufteilen
            print(parts)
            image_name = parts[0]
            print(image_name)
            manufacturer = parts[1]
            print(manufacturer)

            image_path = data_root / "data" / "images" / f"{image_name}.jpg"
            label = classes.index(manufacturer)  # Index des Herstellers als Label
            image = default_loader(str(image_path))
            if transform is not None:
                image = transform(image)
            dataset.append((image, label))
    return dataset

def get_dataloaders(data_root, batch_size=4, num_workers=2):
    transform = transforms.Compose([
        transforms.Resize((500, 500)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    manufacturers_file = data_root / "data" / "manufacturers.txt"
    classes = load_classes(manufacturers_file)

    train_file = data_root / "data" / "images_manufacturer_train.txt"
    test_file = data_root / "data" / "images_manufacturer_test.txt"

    trainset = load_dataset(train_file, data_root, classes, transform)
    testset = load_dataset(test_file, data_root, classes, transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader, len(classes)
