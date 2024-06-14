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

    def __init__(self, pipeline_name):
        self.transforms = None
        self.pipeline_name = pipeline_name
        if hasattr(self, pipeline_name):
            pipeline_call = getattr(self, pipeline_name)
            self.transforms = pipeline_call()
        else:
            raise ValueError("Pipeline name not found")

    @staticmethod
    def default_classification_pipeline_val():
        return v2.Compose(
            [
                v2.Resize(600),
                v2.CenterCrop(size=(600, 600)),
                v2.PILToTensor(),
                # transforms.Grayscale(num_output_channels=3), maybe grayscale ???
                # v2.Grayscale(num_output_channels=1),
                v2.Resize(size=(224, 224)),
                v2.ToDtype(torch.float32),
                # v2.Normalize(
                #     mean=(0, 0, 0), std=(1, 1, 1)
                # ),  # normalize between 0 and 1
                v2.Lambda(lambda x: x / 255.0),
            ]
        )

    # def default_classification_pipeline():
    #     return v2.Compose(
    #         [
    #             TransformsService.default_classification_pipeline_val(),
    #             v2.RandomRotation(degrees=15),
    #             v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25),
    #             v2.RandomHorizontalFlip(),
    #         ]
    #     )

    # TODO: Check if this will yield better results, since original image will be used for training sometimes without extra augmentation
    @staticmethod
    def default_classification_pipeline_train_extra():
        return v2.Compose(
            [
                # TransformsService.default_classification_pipeline_val(),
                v2.RandomRotation(degrees=15),
                v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25),
                v2.RandomHorizontalFlip(),
            ]
        )

    @staticmethod
    def default_classification_pipeline():
        return v2.Compose(
            [
                # TransformsService.default_classification_pipeline_val(),
                v2.RandomChoice(
                    [
                        v2.Compose(
                            [
                                TransformsService.default_classification_pipeline_val(),
                                TransformsService.default_classification_pipeline_train_extra(),
                            ]
                        ),
                        TransformsService.default_classification_pipeline_val(),
                    ],
                    [0.8, 0.2],
                ),
            ]
        )

    @staticmethod
    def bounding_box_base_pipeline():
        """
        Default pipeline for image data
        NOTE = 1 Value in Label:
        1.) Distanz von links -> rechts (innen)
        2.) Distanz von oben -> unten (innen)
        3.) Distanz von links -> rechts (außen)
        4.) Distanz von oben -> unten (außen)

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
                # converts Pil image to tensor as well as pads the image
                PadImageAfter(size=1600),
                # transforms.Grayscale(num_output_channels=3), maybe grayscale ???
                # the label Coordinates have to be scaled by 4 (x/4)
                v2.Resize(size=(400, 400)),
                v2.ToDtype(torch.float32),
                # v2.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),  # normalize between 0 and 1
                # !! ROTATION ALSO TRANSFORM Box Coordinates...!!!!!
                # v2.RandomRotation(degrees=15),  # bounding boxes can not have rotation
                v2.ColorJitter(),
                # !! ALSO TRANSFORM Box Coordinates to Upside Down then...!!!!!
                # v2.RandomHorizontalFlip()
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
                v2.Normalize(
                    mean=(0, 0, 0), std=(1, 1, 1)
                ),  # normalize between 0 and 1
            ]
        )

    @staticmethod
    def scale_coordinates(factor: int):
        """
            Scales the coordinates by a factor

        :param factor: int
        :return: transform
        """
        return v2.Compose([v2.Lambda(lambda x: x / factor)])

    def get_transforms(self):
        return self.transforms


class PadImageAfter:

    def __init__(self, size):
        self.size = size

    def __call__(self, image):  # we assume inputs are always structured like this
        h = image.height
        w = image.width
        #  To Visualize Input Image
        #  image.show()
        img = np.array(image)
        img = np.transpose(img, (2, 0, 1))

        target = np.zeros((3, self.size, self.size))
        target[:, :h, :w] = img
        tensor_img = torch.tensor(target, dtype=torch.uint8)

        """
        # DEBUGGING
        toImgTransform = v2.ToPILImage()
        img = toImgTransform(tensor_img)
        img.show()"""

        return tensor_img
