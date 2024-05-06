"""
    Service Class that executes specific predefined transformations on the data
"""
import torch
from torchvision.transforms import v2


# todo define your own pipeline
class TransformerService:
    def __init__(self, pipeline_array):
        self.pipeline_array = pipeline_array
        # todo based on the pipeline_array, the transformations are executed
        '''
            irgendwie so...
            # Check if the method exists in MyClass and execute it
            if hasattr(my_object, method_name):
            method_to_call = getattr(my_object, method_name)
            method_to_call()
        '''

    def getTransforms(self):
        return self.defaultPipeline

    '''
        Overview of all default transformations:
        https://pytorch.org/vision/stable/transforms.html
    '''

    @staticmethod
    def defaultPipeline():
        """
            Default pipeline for image data
        """
        return v2.Compose([
            v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
            v2.ToDtype(torch.uint8, scale=True),  # optional, most input are already uint8 at this point
            v2.RandomResizedCrop(size=(224, 224), antialias=True),  # Or Resize(antialias=True)
            v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @staticmethod
    def boundig_box_pipeline():
        """
            Pipeline for specific bounding box data
        """
        return v2.Compose([
            v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
            v2.ToDtype(torch.uint8, scale=True),  # optional, most input are already uint8 at this point
            v2.RandomResizedCrop(size=(224, 224), antialias=True),  # Or Resize(antialias=True)
            v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
        ])
