import os

import PIL
import torch
import yaml
from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from yaml import SafeLoader

from src.preprocessing.transforms_service import TransformsService


class Predictor:
    def __init__(self, model, image, device):
        self.model = model
        self.image = image
        self.device = device
        self.parameters = {}
        self.model = model.to(self.device)
        path = os.path.join("..", "..", "data", "config", "base_config.yml")
        with open(path, "r", encoding="utf-8") as stream:
            # Converts yaml document to python object
            self.parameters = yaml.load(stream, Loader=SafeLoader)

    def predict(self):
        self.model.eval()
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        #preprocess = weights.transforms()
        #batch = [preprocess(self.image)]
        transform_pipeline = 'bounding_box_base_pipeline'
        transform_service: TransformsService = TransformsService(transform_pipeline, self.parameters)
        preprocessed_img = transform_service.get_transforms()(self.image).to(self.device)
        prediction = self.model(preprocessed_img)
        labels = [weights.meta["categories"][i] for i in prediction["labels"]]
        box = draw_bounding_boxes(preprocessed_img, boxes=prediction["boxes"],
                                  labels=labels,
                                  colors="red",
                                  width=1, font_size=30)
        im = to_pil_image(box.detach())
        im.show()
        #image = self.image.to(self.device)

    def __call__(self):
        return self.predict()


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
load_model = torch.jit.load(
    "C:\\Projekte\\LearningSoftcomputing\\aircraft-classification\\data\\output\\2024-06-01_14_50_52\\boundingBox_162024_1534_v1.pt")
img_path = "C:\\Users\\admin\\Downloads\\test_1.PNG"
image = PIL.Image.open(img_path).convert("RGB")
predictor = Predictor(load_model, image, device)
predictor()
