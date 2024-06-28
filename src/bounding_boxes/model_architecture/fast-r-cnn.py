import torch
from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from PIL import Image

img_path = "C:\\Projekte\\LearningSoftcomputing\\aircraft-classification\\data\\input\\fgvc-aircraft-2013b\\data\\images\\0043890.jpg"
img = read_image(img_path)

# Step 1: Initialize model with the best available weights
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.90)


class faster_rcnn:
    def __init__(self, model):
        self.model = model
        self.model.train()
        self.model.eval()
        self.weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.preprocess = self.weights.transforms()
        self.batch = [self.preprocess(img)]
        self.prediction = self.model(self.batch)[0]
        self.labels = [self.weights.meta["categories"][i] for i in self.prediction["labels"]]
        self.box = draw_bounding_boxes(img, boxes=self.prediction["boxes"],
                                       labels=self.labels,
                                       colors="red",
                                       width=1, font_size=30)
        self.im = to_pil_image(self.box.detach())
        self.im.show()


# For training

model.train()
images, boxes = torch.rand(4, 3, 600, 1200), torch.rand(4, 11, 4)
boxes[:, :, 2:4] = boxes[:, :, 0:2] + boxes[:, :, 2:4]
labels = torch.randint(1, 91, (4, 11))
images = list(image for image in images)
targets = []
for i in range(len(images)):
    d = {'boxes': boxes[i], 'labels': labels[i]}
    targets.append(d)
    print(images)
    print(targets)
# output = model(images, targets)

"""
# For inference
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = [preprocess(img)]

# Step 4: Use the model and visualize the prediction
prediction = model(batch)[0]
labels = [weights.meta["categories"][i] for i in prediction["labels"]]
print(prediction)
box = draw_bounding_boxes(img, boxes=prediction["boxes"],
                          labels=labels,
                          colors="red",
                          width=1, font_size=30)
im = to_pil_image(box.detach())
im.show()
"""
