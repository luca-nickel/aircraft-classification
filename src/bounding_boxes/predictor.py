import os

import PIL
import cv2
import numpy as np
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
        # self.live_image.show()
        self.device = device
        self.parameters = {}
        self.weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = model.to(self.device)
        path = os.path.join("..", "..", "data", "config", "base_config.yml")
        with open(path, "r", encoding="utf-8") as stream:
            # Converts yaml document to python object
            self.parameters = yaml.load(stream, Loader=SafeLoader)

    def predict(self):
        self.model.eval()
        prediction_transforms = TransformsService(self.parameters)
        # empty tensor to get the offsets out of the transform
        label_offsets = torch.tensor([0, 0, 0, 0], dtype=torch.float32).to(self.device)
        processed_image, label_offset = prediction_transforms.get_prediction_transform()(self.live_image, label_offsets)
        processed_image = processed_image.to(self.device)
        label_offset = label_offset.to(self.device)
        prediction = self.model([processed_image])
        _, prediction = prediction
        prediction = prediction[0]
        prediction_boxes = torch.squeeze(prediction['boxes'])
        true_box_label = self.recalc_bounding_boxes(label_offset, prediction_boxes, self.parameters['rescale_factor'])
        self.draw_bounding_boxes(self.live_image, true_box_label)

    def __call__(self):
        return self.predict()

    @staticmethod
    def recalc_bounding_boxes(label_offset_tensor, boxes, scale_factor=2):
        original_box_labels = boxes - label_offset_tensor
        original_box_labels = original_box_labels * scale_factor
        original_box_labels = original_box_labels.cpu()
        return original_box_labels.detach().numpy()

    @staticmethod
    def draw_bounding_boxes(pil_img, box_coords):
        top_left = (int(box_coords[0]), int(box_coords[1]))
        bottom_right = (int(box_coords[2]), int(box_coords[3]))
        image_np = np.array(pil_img)
        # Convert PIL image to NumPy array (OpenCV format)

        # Convert RGB to BGR (OpenCV uses BGR format)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Draw the bounding box
        cv2.rectangle(image_cv, top_left, bottom_right, color=(255, 255, 0), thickness=2)

        # Convert BGR back to RGB (if needed to show with PIL)
        image_np = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        pil_image_with_box = PIL.Image.fromarray(image_np)
        pil_image_with_box.show()
        return pil_image_with_box


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
