# Import necessary libraries
import tarfile
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random
from pathlib import Path
from torchvision.datasets import FGVCAircraft

# Define a transformation pipeline to apply to the images
# Resize all images to 500x500, convert them to tensors and normalize them
transform = transforms.Compose(
    [transforms.Resize((500, 500)),  # Resize the image to 500x500 pixels
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Define the batch size for the data loader
batch_size = 4

# Load the FGVCAircraft dataset from the specified path and apply the transformations
dataset = FGVCAircraft(Path("../../data"), download=False, transform=transform)

# Create a data loader for the dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

# Define a function to display an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Main execution
if __name__ == '__main__':
    # Open the file containing the training images and their manufacturers
    with open('../../data/fgvc-aircraft-2013b/data/images_manufacturer_train.txt', 'r') as file:
        # Read the file and split it into lines
        lines = file.read().splitlines()
        # Split each line into its components (image index and manufacturer)
        test_indexes_manufacturers = [line.split() for line in lines]

    # Ensure the indexes are valid for the dataset
    valid_indexes = range(len(test_indexes_manufacturers))  # Adjusted to match the length of test_indexes_manufacturers
    
    # Select a random sample of indexes from the valid range
    random_indexes = random.sample(valid_indexes, batch_size)

    # Use the selected random indexes to get the corresponding images and labels
    random_images = [dataset[r_index][0] for r_index in random_indexes]
    labels = [test_indexes_manufacturers[r_index][1] for r_index in random_indexes]

    # Print the labels
    print(labels)

    # Show the images
    imshow(torchvision.utils.make_grid(random_images))