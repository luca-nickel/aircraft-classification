"""
Service Class that executes specific predefined transformations on the data
"""

import torch
from torchvision.transforms import v2


# todo define your own pipeline
class TransformerService:
    """
    Overview of all default transformations:
    https://pytorch.org/vision/stable/transforms.html

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
        """
        return v2.Compose(
            [
                v2.CenterCrop(size=(224, 224)),  # Or Resize(antialias=True)
                v2.PILToTensor(),
                v2.ToDtype(torch.float32),  # Normalize expects float input
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def get_transforms(self):
        return self.transforms
