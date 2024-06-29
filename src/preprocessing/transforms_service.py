"""
    Service Class that executes specific predefined transformations on the data
"""
import numpy as np
import torch
from torchvision.transforms import v2


# todo define your own pipeline
class TransformsService:
    """
    Overview of all default transformations:
    https://pytorch.org/vision/stable/transforms.html

    Write you own transform:
    https://pytorch.org/vision/main/auto_examples/transforms/plot_custom_transforms.html#sphx-glr-auto-examples-transforms-plot-custom-transforms-py
    """

    def __init__(self, function_name_transform, function_name_transforms, function_name_target_transforms,
                 config_parameters=None):
        self.transforms = None
        self.function_name_transform = function_name_transform
        self.function_name_transforms = function_name_transforms
        self.function_name_target_transforms = function_name_target_transforms
        self.config_parameters = config_parameters

    @staticmethod
    def default_classification_pipeline():
        return v2.Compose(
            [
                v2.CenterCrop(size=(600, 600)),
                v2.PILToTensor(),
                # transforms.Grayscale(num_output_channels=3), maybe grayscale ???
                v2.Resize(size=(224, 224)),
                v2.ToDtype(torch.float32),
                v2.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),  # normalize between 0 and 1
                v2.RandomRotation(degrees=15),
                v2.ColorJitter(),
                v2.RandomHorizontalFlip()
            ]
        )

    def transforms_pipeline(self):
        """
            as only transforms ( for data and label, or target/transform(for label) and for data separately)
            can be applied..
        :return:
        """
        return v2.Compose(
            [
                # converts Pil image to tensor as well as pads the image
                pad_image_into_random_position(size=self.config_parameters["image_size"]),
                rescale_image_label(self.config_parameters['rescale_factor']),
                v2.ToDtype(torch.float32),
                color_jitter_on_image(),
                normalize_image()
            ]
        )

    @staticmethod
    def test_transform():
        """
            A pipeline without Augmentation for model testing

        :return: transform
        """
        return v2.Compose(
            [
                v2.ToDtype(torch.float32),
                v2.PILToTensor(),
                v2.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),  # normalize between 0 and 1
            ]
        )

    def get_transforms(self):
        return self.transforms_pipeline()


class normalize_image:
    """
        Normalizes the image
    """

    def __init__(self):
        pass

    def __call__(self, image, label):  # we assume inputs are always structured like this
        """
        For the transformation the image as well as the label have to be adjusted

        :param image:
        :param label:
        :return:
        """
        # Normalize the image
        image = v2.functional.normalize(image, mean=[0, 0, 0], std=[1, 1, 1])

        return image, label


class color_jitter_on_image:
    """
        Applies color jitter on the image
    """

    def __init__(self):
        pass

    def __call__(self, image, label):  # we assume inputs are always structured like this
        """
        For the transformation the image as well as the label have to be adjusted

        :param image:
        :param label:
        :return:
        """
        # Apply color jitter on the image
        image = v2.functional.color_jitter(image)

        return image, label


class rescale_image_label:
    """
        Scales the image and the label by a factor
    """

    def __init__(self, factor):
        self.factor = factor

    def __call__(self, image, label):  # we assume inputs are always structured like this
        """
        For the transformation the image as well as the label have to be adjusted

        :param image:
        :param label:
        :return:
        """
        # Scale the image
        image = v2.functional.rescale(image, scale=self.factor)

        # Scale the label
        label = label / self.factor

        return image, label


class pad_image_into_random_position:
    """
        Pads the image to given size and positions the image at a random position within the padded image
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, image, label):  # we assume inputs are always structured like this
        """
        For the transformation the image as well as the label have to be adjusted

        :param image:
        :param label:
        :return:
        """
        # Define the dimensions of the source and the image
        h = image.height
        w = image.width

        img = np.array(image)
        img = np.transpose(img, (2, 0, 1))  # img = (3, h, w)

        # Create the source array with random values
        source_array = np.random.randint(0, 256, size=(3, self.size, self.size), dtype=np.uint8)

        # Calculate the maximum starting points
        # size because image always square
        max_height = self.size - h
        max_width = self.size - w

        # Generate random starting points
        start_y = np.random.randint(0, max_height)
        start_x = np.random.randint(0, max_width)

        # Insert the image into the source array at the random position
        source_array[:, start_y:start_y + h, start_x:start_x + w] = img
        tensor_img = torch.tensor(source_array, dtype=torch.uint8)

        # DEBUGGING
        toImgTransform = v2.ToPILImage()
        img = toImgTransform(tensor_img)
        img.show()

        # Adjust the label postion of the bounding box
        label_neu = torch.zeros_like(label)
        label_neu[0] = start_x + label[0]  # abstand linker bild rand bis erstes flugzeugteil
        label_neu[1] = start_y + label[1]  # abstand oberer bildrand bis erstes flugzeug
        label_neu[2] = start_x + label[2]  # abstand linker bild rand bis letztes flugzeugteil
        label_neu[3] = start_y + label[3]  # abstand oberer bildrand bis letztes flugzeug

        return tensor_img, label_neu
