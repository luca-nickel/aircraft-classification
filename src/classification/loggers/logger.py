from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime
from PIL import Image
import numpy as np
import os
import torchvision
import torchvision.transforms.functional as TF
import torchvision.utils as TU
from torchvision.transforms import ToTensor
import torch.nn as nn
import matplotlib.pyplot as plt
import io
import torch
import sys


class Logger:

    def __init__(self, name="", locdir="runs", time_stamp=True, step=0):
        self._name = name
        if time_stamp:
            self._dirname = (
                self._name
                + (" - " if name != "" else "")
                + str(datetime.now().strftime("%Y-%m-%d %H_%M"))
            )

        self._locdir = locdir
        self._logger = SummaryWriter(locdir + "/" + self._dirname)
        self._step = step

        self._LOSTAGG = "loss/"
        self._TRAINLOSSTAG = self._LOSTAGG + "train"
        self._VALLOSSTAG = self._LOSTAGG + "val"

        self._ACCTAG = "acc/"
        self._TRAINACCTAG = self._ACCTAG + "train"
        self._VALACCTAG = self._ACCTAG + "val"

        layout = {
            "Losses": {
                "combined_loss": ["Multiline", [self._TRAINLOSSTAG, self._VALLOSSTAG]]
            },
        }
        self._logger.add_custom_scalars(layout)

    def set_step(self, step):
        self._step = step

    def log_string(self, desc, text):
        self._logger.add_text(desc, text)

    def train_start(self):
        print("[LOG]: Training startet.")
        self._trainstart = datetime.now()
        self._logger.add_text("trainduration/start", str(self._trainstart))

    def train_end(self, reason):
        trainend = datetime.now()
        self._logger.add_text("trainduration/end", str(trainend))
        self._logger.add_text(
            "trainduration/duration", str(trainend - self._trainstart)
        )
        self._logger.add_text("trainduration/reason", reason)
        print(
            "[LOG]: Training finished! Runtime {}, because of {}".format(
                (trainend - self._trainstart), reason
            )
        )

    def val_loss(self, value):
        print("[LOG]: Validation Step {} logged. Loss {}".format(self._step, value))
        self._logger.add_scalar(self._VALLOSSTAG, value, self._step)

    def val_acc(self, value):
        print("[LOG]: Validation Step {} logged. Acc {}".format(self._step, value))
        self._logger.add_scalar(self._VALACCTAG, value, self._step)

    def train_loss(self, value):
        print("[LOG]: Training Step {} logged. Loss {}".format(self._step, value))
        self._logger.add_scalar(self._TRAINLOSSTAG, value, self._step)

    def train_acc(self, value):
        print("[LOG]: Training Step {} logged. Acc {}".format(self._step, value))
        self._logger.add_scalar(self._TRAINACCTAG, value, self._step)

    def model_log(self, model, input_data=None):
        model.eval()
        with torch.no_grad():
            self._logger.add_graph(model, input_data, False, False)

    def model_text(self, model):
        self._logger.add_text("model", str(model))

    def summary(self, category, desc):
        self._logger.add_text(
            "summary" + "/" + category, "<pre>" + str(desc) + "</pre>"
        )

    def save_cur_image(self, model, data_input, data_output):
        h = 0.02
        x_min, x_max = data_input[:, 0].min() - 1, data_input[:, 0].max() + 1
        y_min, y_max = data_input[:, 1].min() - 1, data_input[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, h, dtype="float32"),
            np.arange(y_min, y_max, h, dtype="float32"),
        )
        Z = model.predict(torch.tensor(np.c_[xx.ravel(), yy.ravel()]))
        Z = np.argmax(Z.detach().cpu().numpy(), axis=1)
        Z = Z.reshape(xx.shape)
        fig = plt.figure()
        plt.contourf(xx, yy, Z, alpha=0.8)
        plt.scatter(data_input[:, 0], data_input[:, 1], c=data_output, s=40)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = Image.open(buf)
        image = ToTensor()(image)

        self._logger.add_image("image/class", image, self._step)

    def get_model_save_location(self):
        path = os.path.join(self._locdir, self._dirname, "model_saves")
        if not os.path.exists(path):
            os.makedirs(path)
        return os.path.join(
            path, self._name + "_" + "{:03d}".format(self._step) + "_" + ".pt"
        )

    def close(self):
        print("[LOG]: Closing logger.")
        self._logger.close()
