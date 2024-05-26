"""
    Service Class that executes specific predefined transformations on the data
"""

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

    def __init__(self, pipeline_name):
        self.transforms = None
        self.pipeline_name = pipeline_name
        if hasattr(self, pipeline_name):
            pipeline_call = getattr(self, pipeline_name)
            self.transforms = pipeline_call()
        else:
            raise ValueError("Pipeline name not found")

    @staticmethod
    def bounding_box_base_pipeline():
        """
            Default pipeline for image data

            Idea:
            Generate fixed sized images, with different random augmentations so increasing the training iterationn
            equals an increase of the dataset
            incorporates Data Augmentation, the number of epochs how to be increased as the augmentation is random!

            @aykan
            Important Note:
            CenterCrop has different behaviour for PIL and tensor image. In case of PIL image, it pads the image,
            while for tensor, it crops incorrectly[Checked code, it just computes offset(which are negative) and uses
            tensor indexing to extract the crop area which is incorrect.

            Concept:
            1. Pad the image so all images have the same size (1400x1400)
            2. Resize the image to For Example 350x350 (1400/4)
            3. Do the Augmentation
            4. resize the detected coordinates * 4

        """
        return v2.Compose(
            [
                v2.CenterCrop(size=(1400, 1400)),
                #  transforms.Grayscale(num_output_channels=3), maybe grayscale ???
                v2.PILToTensor(),
                v2.Resize(size=(350, 350)),
                v2.ToDtype(torch.float32),
                #v2.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),  # normalize between 0 and 1
                #v2.RandomRotation(degrees=15), bounding boxes can not have rotation
                v2.ColorJitter(),
                v2.RandomHorizontalFlip()
            ]
        )

    @staticmethod
    def transform_tensor_to_img():
        """
            A pipeline to transform tensor to PIL Image

        :return: transform
        """
        return v2.Compose(
            [
                v2.ToPILImage()
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
        return self.transforms
