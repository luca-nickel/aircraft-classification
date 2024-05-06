import os
from datetime import datetime

import torch
from torchvision.datasets import FGVCAircraft

from src.bounding_boxes.model_architecture.CNN_bounding_boxes_architecture import CNN_model_bounding_boxes
from src.logging.export_Service import ExportService
from src.preprocessing.transformerService import TransformerService
from trainer import model_trainer


class bounding_box_training_exe:
    def __init__(self, parameters):
        self.parameters = parameters

    def exe(self):
        now = datetime.now()
        begin_time = now.strftime("%Y-%m-%d_%H_%M_%S")
        print("Beginn Time: " + begin_time)
        EXPORT_PATH = os.path.join(self.parameters['RESULT_FOLDER'], begin_time)
        exporter: ExportService = ExportService(EXPORT_PATH)
        transformPipeline: list = self.parameters['DATA_AUGMENTATION']
        transformService: TransformerService = TransformerService(transformPipeline)
        dataset_train = FGVCAircraft(root=self.parameters['DATASET_PATH'], download=False,
                                     transform=transformService.getTransforms())
        len_features = len(dataset_train)
        len_tsl = len(dataset_train)
        model = CNN_model_bounding_boxes(len_tsl, len_features)
        #loss_func = torch.nn.CrossEntropyLoss()  # classyfi
        loss_func = torch.nn.MSELoss()  # MSE
        LR = self.parameters['LR']
        L2RegularisationFactor = self.parameters['L2RegularisationFactor']
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=L2RegularisationFactor)
        trainer = model_trainer(self.parameters, dataset_train, model, loss_func, optimizer)
        model = trainer.run(exporter)
        exporter.storeModel(model, self.parameters['MODEL_NAME'])