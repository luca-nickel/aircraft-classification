import os
from math import ceil
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import tensorflow as tf
import torch
import torch.nn as nn
from torchvision import transforms
import xplique
from xplique.wrappers import TorchWrapper
from xplique.plots import plot_attributions

tf.config.run_functions_eagerly(True)

img_list = [
    ('airplane1.jpg', 404),
    ('airplane2.jpg', 404),
    ('airplane3.jpg', 404),
    ('airplane4.jpg', 404),
    ('airplane5.jpg', 404)
]

"""
Given a numpy array, extracts the largest possible square and resizes it to
the requested size
"""
def central_crop_and_resize(img, size=224):

  h, w, _ = img.shape

  min_side = min(h, w)
  max_side_center = max(h, w) // 2.0

  min_cut = int(max_side_center-min_side//2)
  max_cut = int(max_side_center+min_side//2)

  img = img[:, min_cut:max_cut] if w > h else img[min_cut:max_cut]
  img = tf.image.resize(img, (size, size))

  return img

X = []
Y = []

for img_name, label in img_list:
    img = cv2.imread(img_name)[..., ::-1] # when cv2 load an image, the channels are inversed
    img = central_crop_and_resize(img)
    label = tf.keras.utils.to_categorical(label, 1000)

    X.append(img)
    Y.append(label)

X = np.array(X, dtype=np.uint8) # slight change here
Y = np.array(Y)

plt.rcParams["figure.figsize"] = [15, 6]
for img_id, img in enumerate(X):
  plt.subplot(1, len(X), img_id+1)
  plt.imshow(img)
  plt.axis('off')

# load model
model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True, verbose=False)
model.eval()

# add the preprocessing function, as resizing and cropping is already applied
# we do not add them in the compose
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

X_preprocessed = torch.stack([preprocess(x) for x in X])

# make prediction
with torch.no_grad():
    output = model(X_preprocessed)

probabilities = torch.nn.functional.softmax(output, dim=1)

# labels
os.system("wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt")

# Read the categories
with open("imagenet_classes.txt", "r") as f:
    classes = [s.strip() for s in f.readlines()]
# Get label prediction for each image
preds = torch.argmax(probabilities, dim=1)
preds = preds.detach().numpy()
preds_labels = np.array([classes[pred] for pred in preds])

# Assert the predictions are right
for img_id, img in enumerate(X):
  plt.subplot(1, len(X), img_id+1)
  plt.title(f"{preds_labels[img_id]}: {preds[img_id]}")
  plt.imshow(img)
  plt.axis('off')

from xplique.attributions import (Saliency, GradientInput, IntegratedGradients, SmoothGrad, VarGrad,
                                  SquareGrad, Occlusion, Rise, GuidedBackprop, Lime, KernelShap, SobolAttributionMethod)

# wrap the torch model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
wrapped_model = TorchWrapper(model, device)

# modify inputs to match explainers expectations
X_preprocessed4explainer = np.moveaxis(X_preprocessed.numpy(), [1, 2, 3], [3, 1, 2])


"""## Explain inputs"""

# set batch size parameter
batch_size = 64

# build the explainers
explainers = [
             Saliency(wrapped_model),
             GradientInput(wrapped_model),
             IntegratedGradients(wrapped_model, steps=80, batch_size=batch_size),
             SmoothGrad(wrapped_model, nb_samples=80, batch_size=batch_size),
             SquareGrad(wrapped_model, nb_samples=80, batch_size=batch_size),
             #VarGrad(wrapped_model, nb_samples=80, batch_size=batch_size),
             Occlusion(wrapped_model, patch_size=10, patch_stride=5, batch_size=batch_size),
             #Rise(wrapped_model, nb_samples=4000, batch_size=batch_size),
             #SobolAttributionMethod(wrapped_model, batch_size=batch_size),
              Lime(wrapped_model, nb_samples = 4000, batch_size=batch_size),
              KernelShap(wrapped_model, nb_samples = 4000, batch_size=batch_size)
]

for explainer in explainers:

  explanations = explainer(X_preprocessed4explainer, Y)

  print(f"Method: {explainer.__class__.__name__}")
  plot_attributions(explanations, X, img_size=2., cmap='jet', alpha=0.4,
                    cols=len(X), absolute_value=True, clip_percentile=0.5)
  plt.show()
  print("\n")