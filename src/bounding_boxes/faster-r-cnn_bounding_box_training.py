import os
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2

from yaml import SafeLoader
from datetime import datetime
from src.bounding_boxes.fgvs_aircraft_custom_dataset import FgvcAircraftBbox
from src.bounding_boxes.model_architecture.cnn_bounding_boxes_architecture import (
    CnnModelBoundingBoxes,
)
from src.logging.export_service import ExportService
from src.preprocessing.transforms_service import TransformsService
from src.trainer import ModelTrainer


class BoundingBoxTraining:
    def __init__(self, model):
        self.parameters = {}
        self.model = model
        path = os.path.join("..", "..", "data", "config", "base_config.yml")
        with open(path, "r", encoding="utf-8") as stream:
            # Converts yaml document to python object
            self.parameters = yaml.load(stream, Loader=SafeLoader)

    def run(self):
        now = datetime.now()
        begin_time = now.strftime("%Y-%m-%d_%H_%M_%S")
        print("Beginn Time: " + begin_time)
        export_path = os.path.join(self.parameters["result_folder"], begin_time)
        exporter: ExportService = ExportService(export_path)
        transform_pipeline: list = self.parameters["data_augmentation_pipeline"]
        transform_service: TransformsService = TransformsService(transform_pipeline)
        dataset_train = FgvcAircraftBbox(
            root=self.parameters["dataset_path"],
            file="images_bounding_box_train.txt",
            download=False,
            transform=transform_service.get_transforms(),
            target_transform=transform_service.scale_coordinates(4)
        )
        dataset_test = FgvcAircraftBbox(
            root=self.parameters["dataset_path"],
            file="images_bounding_box_test.txt",
            download=False,
            transform=transform_service.get_transforms(),
            target_transform=transform_service.scale_coordinates(4)
        )
        train_dataloader = DataLoader(
            dataset=dataset_train,
            batch_size=self.parameters["batch_size"],
            shuffle=True,
            num_workers=0,
        )
        test_dataloader = DataLoader(
            dataset=dataset_test,
            batch_size=self.parameters["batch_size"],
            shuffle=True,
            num_workers=0,
        )
        len_tsl = len(dataset_train)
        print(len_tsl)

        self.model.train()
        writer = SummaryWriter(
            os.path.join(
                "logs",
                f"{begin_time}"
            )
        )
        # returns batches
        for i, z in enumerate(train_dataloader):
            images = z[0]
            targets = []
            c = 0
            for image in images:
                box = z[1][c]
                box = box.unsqueeze(0)
                label = torch.tensor(1, dtype=torch.int64)
                label = label.unsqueeze(0)
                # targets supposed to be a list of dicts with keys 'boxes' and 'labels'
                d = {'boxes': box, 'labels': label}
                targets.append(d)
                c += 1
            output = self.model(images, targets)
            print(output)

        exporter.store_model(self.model, self.parameters["model_name"])


###
print(torch.cuda.is_available())
###
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model_faster = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.90)
model_faster.train()
exe = BoundingBoxTraining(model_faster)
exe.run()
