import os
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
from torchvision.transforms import v2

from yaml import SafeLoader
from datetime import datetime
from src.bounding_boxes.fgvs_aircraft_custom_dataset import FgvcAircraftBbox
from src.bounding_boxes.model_architecture.cnn_bounding_boxes_architecture import (
    CnnModelBoundingBoxes,
)
from src.logging.export_service import ExportService
from src.preprocessing.transforms_service import TransformsService
from src.trainer import ModelTrainer
import torch.nn.functional as F

###
print(torch.cuda.is_available())
###
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(device)


class BoundingBoxTraining:
    def __init__(self, model):
        self.parameters = {}
        self.model = model.to(device)
        path = os.path.join("..", "..", "data", "config", "base_config.yml")
        with open(path, "r", encoding="utf-8") as stream:
            # Converts yaml document to python object
            self.parameters = yaml.load(stream, Loader=SafeLoader)

        self.now = datetime.now()
        self.begin_time = self.now.strftime("%Y-%m-%d_%H_%M_%S")
        print("Beginn Time: " + self.begin_time)
        self.export_path = os.path.join(self.parameters["result_folder"], self.begin_time)
        self.exporter: ExportService = ExportService(self.export_path)
        self.transform_pipeline: list = self.parameters["function_name_transform"]
        self.transforms_pipeline: list = self.parameters["function_name_transforms"]
        self.target_transform_pipeline: list = self.parameters["function_name_target_transforms"]
        self.transform_service: TransformsService = TransformsService(self.transform_pipeline,
                                                                      self.transforms_pipeline,
                                                                      self.target_transform_pipeline,
                                                                      self.parameters)
        self.dataset_train = FgvcAircraftBbox(
            root=self.parameters["dataset_path"],
            file="images_bounding_box_train.txt",
            download=False,
            transforms=self.transform_service.get_transforms(),
            transform=None,
            target_transform=None
        )
        self.dataset_test = FgvcAircraftBbox(
            root=self.parameters["dataset_path"],
            file="images_bounding_box_test.txt",
            download=False,
            transforms=self.transform_service.get_transforms(),
            transform=None,
            target_transform=None
        )
        self.train_dataloader = DataLoader(
            dataset=self.dataset_train,
            batch_size=self.parameters["batch_size"],
            shuffle=True,
            num_workers=0,
        )
        self.test_dataloader = DataLoader(
            dataset=self.dataset_test,
            batch_size=self.parameters["batch_size"],
            shuffle=True,
            num_workers=0,
        )
        self.writer = SummaryWriter(
            os.path.join(
                "logs",
                f"{self.begin_time}"
            )
        )
        len_tsl = len(self.dataset_train)
        print(len_tsl)

    def run(self):
        log_idx = 0
        epochs = self.parameters["num_epoch"]
        for epoch in range(epochs):
            print(f"Epoch: {epoch}")
            log_idx += self.train_epoch(log_idx)
            self.test_model(epoch)

    def train_epoch(self, log_idx):
        self.model.train()
        # returns batches
        for i, z in enumerate(self.train_dataloader):
            images = z[0].to(device)
            targets = []
            c = 0
            for image in images:
                # To Visualize Input Image
                toImgTransform = v2.ToPILImage()
                img = toImgTransform(image)
                img.show()

                box = z[1][c].to(device)
                box = box.unsqueeze(0)
                # In COCO DateSet, which faste r-cnn was trained on label=5 is airplane
                label = torch.tensor(5, dtype=torch.int64).to(device)
                label = label.unsqueeze(0)
                # targets supposed to be a list of dicts with keys 'boxes' and 'labels'
                d = {'boxes': box, 'labels': label}
                targets.append(d)
                c += 1

            output = self.model(images, targets)
            self.writer.add_scalar("Loss/train", output['loss_box_reg'].item(), log_idx)
            log_idx += 1
            if i % 5 == 0:
                print(f"Batch: {i}")

            return log_idx

        self.exporter.store_model(self.model, self.parameters["model_name"])

    def test_model(self, epoch):
        self.model.eval()
        counter = 0
        for i, z in enumerate(self.test_dataloader):
            images = z[0].to(device)
            targets = []
            c = 0
            box_labels = torch.zeros((len(images), 4))
            for image in images:
                """
                # To Visualize Input Image
                toImgTransform = v2.ToPILImage()
                img = toImgTransform(image)
                img.show()
                """

                box = z[1][c].to(device)
                box_labels[c] = box
                box = box.unsqueeze(0)
                label = torch.tensor(1, dtype=torch.int64)
                label = label.unsqueeze(0).to(device)
                # targets supposed to be a list of dicts with keys 'boxes' and 'labels'
                d = {'boxes': box, 'labels': label}
                targets.append(d)
                c += 1
            counter += 1
            output = self.model(images, targets)
            # Extract the 'boxes' values from each dictionary
            boxes_list = []
            for d in output:
                if len(d['boxes']) > 0:
                    boxes_list.append(d['boxes'][0].to(device))
                else:
                    empty = torch.zeros(4).to(device)
                    boxes_list.append(empty)

            # Stack the list of tensors into a single tensor
            output_tensor = torch.stack(boxes_list, dim=0).to(device)

            loss = F.mse_loss(output_tensor, box_labels.to(device), reduction='mean').to(device)
            self.writer.add_scalar(f"Loss/test_{epoch}", loss.item(), counter)


weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model_faster = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.90)
model_faster.train()
exe = BoundingBoxTraining(model_faster)
exe.run()
