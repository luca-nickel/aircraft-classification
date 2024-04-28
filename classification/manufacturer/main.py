import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random
from pathlib import Path
from torchvision.datasets import FGVCAircraft


transform = transforms.Compose(
    [transforms.Resize((500, 500)),  # Resize all images to 500x500
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

dataset = FGVCAircraft(Path("../../data"), download=True, transform=transform)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == '__main__':
    with open('../../data/fgvc-aircraft-2013b/data/images_manufacturer_train.txt', 'r') as file:
        lines = file.read().splitlines()
        test_indexes_manufacturers = [line.split() for line in lines]
        #print(test_indexes_manufacturers)

    random_elements = random.sample(test_indexes_manufacturers, batch_size)

    random_indexes = [1917860, 225987, 1401747, int('0869722')]

    #random_images = [dataset[int(i[0]) - 1][0] for i in random_elements]
    
    labels = [element[1] for element in random_elements]

    print(labels)
    # Show images
    imshow(torchvision.utils.make_grid(random_images))

    
        

