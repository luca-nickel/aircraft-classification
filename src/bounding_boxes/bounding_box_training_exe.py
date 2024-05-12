import os
import pathlib
from datetime import datetime

import torch

from src.bounding_boxes.fgvs_aircraft_custom_dataset import FGVCAircraft_bbox
from src.bounding_boxes.model_architecture.cnn_bounding_boxes_architecture import cnn_model_bounding_boxes
from src.logging.export_service import export_service
from src.preprocessing.config_loader import config_loader
from src.preprocessing.transformer_service import transformer_service
from trainer import model_trainer


class bounding_box_training_exe:
    def __init__(self):

        self.parameters = config_loader.load_model_config(os.path.join('..', '..', 'data', 'config', 'base_config.yml'))

    def exe(self):
        now = datetime.now()
        begin_time = now.strftime("%Y-%m-%d_%H_%M_%S")
        print("Beginn Time: " + begin_time)
        EXPORT_PATH = os.path.join(self.parameters['result_folder'], begin_time)
        exporter: export_service = export_service(EXPORT_PATH)
        transformPipeline: list = self.parameters['data_augmentation_pipeline']
        transformService: transformer_service = transformer_service(transformPipeline)
        dataset_train = FGVCAircraft_bbox(root=self.parameters['dataset_path'], file="images_bounding_box_train.txt",
                                          download=False, transform=transformService.get_transforms())
        dataset_test = FGVCAircraft_bbox(root=self.parameters['dataset_path'], file="images_bounding_box_test.txt",
                                         download=False, transform=transformService.get_transforms())
        len_tsl = len(dataset_train)
        print(len_tsl)
        model = cnn_model_bounding_boxes(224)
        # loss_func = torch.nn.CrossEntropyLoss()  # classyfi
        loss_func = torch.nn.MSELoss()  # MSE
        LR = self.parameters['lr']
        L2RegularisationFactor = self.parameters['l2_regularisation_factor']
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=L2RegularisationFactor)
        trainer = model_trainer(self.parameters, dataset_train, dataset_test, model, loss_func, optimizer, exporter)
        model = trainer.run()
        exporter.store_model(model, self.parameters['model_name'])


exe = bounding_box_training_exe()
print(torch.cuda.is_available())
exe.exe()
