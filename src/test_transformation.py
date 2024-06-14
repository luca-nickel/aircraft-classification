from preprocessing.transforms_service import TransformsService
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
from random import choice


def main():

    images = glob("../data/input/fgvc-aircraft-2013b/data/images/*.jpg")

    for i in range(10):
        image = Image.open(choice(images)).convert("RGB")
        transform = TransformsService("default_classification_pipeline").transforms
        transformed_image = transform(image)

        # Convert the tensor image to numpy for displaying
        numpy_image = transformed_image.numpy().transpose((1, 2, 0))

        # Display the image
        plt.imshow(numpy_image)
        plt.show()


if __name__ == "__main__":
    main()
