import os
from datetime import datetime
from logging.export_service import ExportService

import torch

from src.bounding_boxes.fgvs_aircraft_custom_dataset import FGVCAircraft_bbox
from src.bounding_boxes.model_architecture.CNN_bounding_boxes_architecture import (
    CNN_model_bounding_boxes,
)
from src.preprocessing.config_loader import ConfigLoader
from src.preprocessing.transformer_service import TransformerService
from trainer import model_trainer


class bounding_box_training_exe:
    def __init__(self):
        self.parameters = ConfigLoader.load_model_config(
            "C:\\Projekte\\LearningSoftcomputing\\aircraft-classification\\data\\config\\base_config.yml"
        )

    def exe(self):
        now = datetime.now()
        begin_time = now.strftime("%Y-%m-%d_%H_%M_%S")
        print("Beginn Time: " + begin_time)
        EXPORT_PATH = os.path.join(self.parameters["RESULT_FOLDER"], begin_time)
        exporter: ExportService = ExportService(EXPORT_PATH)
        transformPipeline: list = self.parameters["DATA_AUGMENTATION_PIPELINE"]
        transformService: TransformerService = TransformerService(transformPipeline)
        dataset_train = FGVCAircraft_bbox(
            root=self.parameters["DATASET_PATH"],
            file="images_bounding_box_train.txt",
            download=False,
            transform=transformService.get_transforms(),
        )
        dataset_test = FGVCAircraft_bbox(
            root=self.parameters["DATASET_PATH"],
            file="images_bounding_box_test.txt",
            download=False,
            transform=transformService.get_transforms(),
        )
        len_tsl = len(dataset_train)
        print(len_tsl)
        model = CNN_model_bounding_boxes(224)
        # loss_func = torch.nn.CrossEntropyLoss()  # classyfi
        loss_func = torch.nn.MSELoss()  # MSE
        LR = self.parameters["LR"]
        L2RegularisationFactor = self.parameters["L2RegularisationFactor"]
        optimizer = torch.optim.Adam(
            model.parameters(), lr=LR, weight_decay=L2RegularisationFactor
        )
        trainer = model_trainer(
            self.parameters, dataset_train, dataset_test, model, loss_func, optimizer
        )
        model = trainer.run(exporter)
        exporter.store_model(model, self.parameters["MODEL_NAME"])


exe = bounding_box_training_exe()
print(torch.cuda.is_available())
exe.exe()
