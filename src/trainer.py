import os

import torch
from torch.utils.data import DataLoader
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from src.logging.export_service import ExportService
from src.preprocessing.transforms_service import TransformsService
from torchvision.transforms import transforms as v2

# Check for Gpu
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

device = torch.device("cpu")
print("Device: ", device)


class ModelTrainer:

    def __init__(
            self,
            parameters,
            tr_dataset,
            test_dataset,
            model,
            loss_func,
            optimizer,
            exporter: ExportService,
    ):
        now = datetime.now()
        start_time = now.strftime("%Y-%m-%d_%H_%M_%S")
        self.exporter = exporter
        self.parameters = parameters
        self.logging_interval = parameters["logging_interval"]
        self.num_epoch = parameters["num_epoch"]
        self.lr = parameters["lr"]
        self.batch_size = parameters["batch_size"]
        self.l2_regularisation_factor = parameters["l2_regularisation_factor"]
        self.dataset_train = tr_dataset
        self.train_dataloader = DataLoader(
            dataset=self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )
        self.dataset_test = test_dataset
        self.test_dataloader = DataLoader(
            dataset=self.dataset_test,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )
        # todo fix some name to identify run

        self.writer = SummaryWriter(
            os.path.join(
                "logs",
                f"{start_time}_lr_{self.lr}_batch_size_{self.batch_size}_l2_"
                f"{self.l2_regularisation_factor}",
            )
        )

        self.model = model
        model.to(device)
        self.loss = loss_func
        self.optimizer = optimizer

    def run(self):
        for e in range(self.num_epoch):
            self.model.train()
            for i, z in enumerate(self.train_dataloader):
                x = z[0]

                """
                # To Visualize Input Image
                toImgTransform = v2.ToPILImage()
                tensorImg = x[0]
                img = toImgTransform(tensorImg)
                img.show()
                """

                #  test show image
                y = z[1]
                loss_val, model_out = self.training_step(
                    self.model, x, y, self.loss, self.optimizer
                )
                self.writer.add_scalar("Loss/train", loss_val, i)
                if i % self.logging_interval == 0:
                    # Logging Single EPOCH TRAIN

                    if i % 2 == 0:
                        print("Epoch: {}, Batch: {}".format(e + 1, i))
                        # '''
                        print("IN INTERVAL LOGGING")
                        print("model_INPUT:")
                        print(x)
                        print("model_out")
                        print(model_out)
                        print("y")
                        print(y)
                        print("Training Loss: {}".format(loss_val))
                        print("________________")
                        # '''
            # soll-ist logging training epoch

            ##### TEST FOR EACH EPOCH
            with torch.no_grad():
                epochTestArr = self.test_model(
                    self.test_dataloader, self.model, self.loss
                )
                print(epochTestArr)

        now = datetime.now()
        end_time = now.strftime("%Y-%m-%d_%H_%M_%S")
        print("End Time =", end_time)
        return self.model

    def test_model(self, test_data, model, loss_func):
        with torch.no_grad():
            # Model should not improve
            model.eval()
            for i, z in enumerate(test_data):
                inputs = z[0]
                y = z[1]
                inputs = inputs.to(device)
                y = y.to(device)
                # check each item in batch
                # eigtl nunnötig da Batch_size = 1
                for j in range(len(y)):
                    model_out = model(inputs.float())
                    model_out = torch.squeeze(model_out, 1)
                    y_val = y[j].float()
                    # bei anderer Fehlerfunktion zu ändern, bzw. über loss.item() definieren
                    loss_val = loss_func(model_out, y_val)
                    model_out_val = model_out[j]
                # todo: add logging hyper param under paramter/*
                self.writer.add_scalar("Loss/test", loss_val, i)

        return [model_out_val, loss_val]

    @staticmethod
    def training_step(model, x, y, loss, optimizer):
        x = x.to(device)
        y = y.to(device)
        model_out = model(x.float())
        model_out = torch.squeeze(model_out, 1)
        # to put model out in same shape as y
        optimizer.zero_grad()
        # MSE
        loss_val = loss(model_out, y.float())
        loss_val.backward()
        optimizer.step()
        return loss_val, model_out
