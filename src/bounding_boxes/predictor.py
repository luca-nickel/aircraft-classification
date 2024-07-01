import os

import PIL
import torch
import yaml
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes
from yaml import SafeLoader
from torchvision.transforms import v2
from src.preprocessing.transforms_service import TransformsService


class Predictor:
    def __init__(self, model, live_image, device):
        self.model = model
        self.live_image = live_image
        #self.live_image.show()
        self.device = device
        self.parameters = {}
        self.model = model.to(self.device)
        path = os.path.join("..", "..", "data", "config", "base_config.yml")
        with open(path, "r", encoding="utf-8") as stream:
            # Converts yaml document to python object
            self.parameters = yaml.load(stream, Loader=SafeLoader)

    def predict(self):
        self.model.eval()
        prediction_transforms = TransformsService(self.parameters)
        processed_image = prediction_transforms.get_prediction_transform()(self.live_image).to(self.device)
        # debug_img = to_pil_image(processed_image)
        # debug_img.show()
        prediction = self.model([processed_image])
        print(prediction)
        # labels = [weights.meta["categories"][i] for i in prediction["labels"]]
        _, prediction_boxes = prediction
        prediction_boxes = torch.squeeze(prediction_boxes[0]['boxes'])
        tensor_live_img = v2.functional.to_tensor(self.live_image)
        box = draw_bounding_boxes(tensor_live_img,
                                  boxes=prediction_boxes,
                                  labels=['aircraft'],
                                  colors="red",
                                  width=1, font_size=30)
        im = to_pil_image(box.detach())
        im.show()

    def __call__(self):
        return self.predict()


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
load_model = torch.jit.load(
    "C:\\Projekte\\LearningSoftcomputing\\aircraft-classification\\src\\bounding_boxes\\output\\results\\2024-06"
    "-30_23_55_15\\base_test172024_042.pt")
img_path = "C:\\Projekte\\LearningSoftcomputing\\aircraft-classification\\data\\input\\fgvc-aircraft-2013b\\data" \
           "\\images\\0056589.jpg"
image = PIL.Image.open(img_path).convert("RGB")
predictor = Predictor(load_model, image, device)
predictor()
