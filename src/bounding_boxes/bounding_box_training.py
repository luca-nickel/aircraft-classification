import os
from datetime import datetime

import torch
import yaml
from yaml import SafeLoader

from src.bounding_boxes.fgvs_aircraft_custom_dataset import FgvcAircraftBbox
from src.bounding_boxes.model_architecture.cnn_bounding_boxes_architecture import (
    CnnModelBoundingBoxes,
)
from src.logging.export_service import ExportService
from src.preprocessing.transforms_service import TransformsService
from src.trainer import ModelTrainer


class BoundingBoxTraining:
    def __init__(self):
        self.parameters = {}
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
        )
        dataset_test = FgvcAircraftBbox(
            root=self.parameters["dataset_path"],
            file="images_bounding_box_test.txt",
            download=False,
            transform=transform_service.get_transforms(),
        )
        len_tsl = len(dataset_train)
        print(len_tsl)
        model = CnnModelBoundingBoxes(224)
        # loss_func = torch.nn.CrossEntropyLoss()  # classyfi
        loss_func = torch.nn.MSELoss()  # MSE
        lr = self.parameters["lr"]
        l2_regularisation_factor = self.parameters["l2_regularisation_factor"]
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=l2_regularisation_factor
        )
        trainer = ModelTrainer(
            self.parameters,
            dataset_train,
            dataset_test,
            model,
            loss_func,
            optimizer,
            exporter,
        )
        model = trainer.run()
        exporter.store_model(model, self.parameters["model_name"])


exe = BoundingBoxTraining()
print(torch.cuda.is_available())
exe.run()
