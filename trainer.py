from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.logging.export_service import ExportService

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class ModelTrainer:
    """
    TODO: Needs to be filled by Aykan
    """

    def __init__(
        self, parameters, tr_dataset, test_dataset, model, loss_func, optimizer
    ):
        self.parameters = parameters
        self.logging_interval = parameters["LOGGING_INTERVAL"]
        self.num_epoch = parameters["NUM_EPOCH"]
        self.learning_rate = parameters["LR"]
        self.batch_size = parameters["BATCH_SIZE"]
        self.l2_regularisation_factore = parameters["L2RegularisationFactor"]
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
        self.model = model
        model.to(device)
        self.loss = loss_func
        self.optimizer = optimizer
        self.printer = 100000
        self.log_index_train = 0
        self.log_index_test = 0

    def run(self, exporter: ExportService):
        """
        Ausführung:
            Für jede Epoche:
                * Training (model adjust)
                * Test (model freeze)

        Gespeichert wird:
            Für Jede Epoche:
                * Training:
                    csv: Soll_Werte Array, Ist_Werte Array, laufender loss, laufender durchschnittliche Loss
                    img: soll-ist Vergleich mit Running Mean, laufender loss
                * Test:
                    csv: Soll_Werte Array, Ist_Werte Array, laufender loss, laufender durchschnittliche Loss
                    img: soll-ist Vergleich mit Running Mean, laufender loss
            Für Alle Epochen Zusammen:
                * Training:
                    csv: Soll_Werte Array, Ist_Werte Array, laufender Loss, laufender avg loss --Intervalliert
                    img: soll-ist Vergleich mit Running Mean, laufender loss
                * Test
                    csv: Soll_Werte Array, Ist_Werte Array, laufender Loss, laufender avg loss --Intervalliert
                    img: soll-ist Vergleich mit Running Mean, laufender loss
        """
        # np array mit dim = (x,y) alle x-te trainingschritte werden gelogged, dabei werden y messwerte gespeichert
        # y1 = 'epoch', y2='modelOut', y3='label', y4='loss', y5='running_avg'
        train_logging_arr_interval = np.zeros(
            (
                int(
                    (
                        len(self.train_dataloader)
                        * self.num_epoch
                        / self.logging_interval
                    )
                )
                + self.num_epoch,
                5,
            )
        )
        train_epoch_logging_arr_interval = np.zeros(
            (int((len(self.train_dataloader) / self.logging_interval)) + 2, 5)
        )
        log_columns_train = [
            "EPOCH",
            "idx",
            "rIDX",
            "prediction",
            "label",
            "LOSS",
            "epochAvgLOSS",
            "rAvgLOSS",
        ]
        log_columns_test = [
            "EPOCH",
            "idx",
            "prediction",
            "label",
            "LOSS",
            "epochAvgLOSS",
        ]
        train_logging_arr = np.zeros(
            (
                int((len(self.train_dataloader) * self.num_epoch)) + self.num_epoch,
                len(log_columns_train),
            )
        )
        test_logging_arr = np.zeros((1, len(log_columns_test)))

        rRunningAvgLoss = 0.0
        # running var for each model exec step
        c = 0
        # logging running var
        logIdx_train = 0
        print("LENGTH OF DATALOADER")
        print(len(self.train_dataloader))
        for e in range(self.num_epoch):
            self.model.train()
            eAvgLoss = 0.0
            ############## FOR EPOCH #############
            ######## EPOCH Train #######
            ### ______________ TRAIN ______________ ###
            for i, z in enumerate(self.train_dataloader):
                c = c + 1
                # todo check ?? the label dimensions etc are mixed up if including batch_size??
                x = z[0]
                y = z[1]
                loss_val, model_out = self.training_step(
                    self.model, x, y, self.loss, self.optimizer
                )
                # fix logging are 4 labels now
                # rLabel = round(y.item(), 4)
                # rModel_out = round(model_out.item(), 4)
                rLoss = round(loss_val.item(), 4)
                rRunningAvgLoss = round(((rRunningAvgLoss + rLoss) / c), 4)
                eAvgLoss = round(((eAvgLoss + rLoss) / (i + 1)), 4)
                """
                train_logging_arr = self.createLogEntry_train(train_logging_arr, e, i, c, rModel_out, rLabel, rLoss,
                                                              eAvgLoss,
                                                              rRunningAvgLoss)"""
                train_logging_arr = self.create_log_entry_train(
                    train_logging_arr, e, i, c, 0, 0, rLoss, eAvgLoss, rRunningAvgLoss
                )
                if i % self.logging_interval == 0:
                    # Logging Single EPOCH TRAIN
                    # train_logging_arr_interval[logIdx_train] = [int(e), rModel_out, rLabel, rLoss, rRunningAvgLoss]
                    train_logging_arr_interval[logIdx_train] = [
                        int(e),
                        0,
                        0,
                        rLoss,
                        rRunningAvgLoss,
                    ]
                    logIdx_train = logIdx_train + 1
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
                tmp_test_logging_arr = np.zeros(
                    (int(len(self.test_dataloader) + 1), len(log_columns_test))
                )
                epochTestArr = self.test_model(
                    self.test_dataloader, self.model, self.loss, tmp_test_logging_arr, e
                )
                test_logging_arr = np.concatenate((test_logging_arr, epochTestArr))
                exporter.createLogAndPlot_epoch_test(
                    epochTestArr, e, log_columns_test, "TEST", "test"
                )

        # FÜR GESAMT RESULT
        LOSS_PLOT_INTERVAL = self.parameters["LOSS_PLOT_INTERVAL"]
        exporter.saveAllResults(
            train_logging_arr, log_columns_train, "TRAINING", "ALL_TR_RESULTS"
        )
        exporter.createLossPlot(
            train_logging_arr,
            LOSS_PLOT_INTERVAL,
            log_columns_train,
            "TRAINING",
            "TRAINING_LOGGING_INTERVAL",
        )
        exporter.createLogAndPlot_epoch_test(
            test_logging_arr,
            self.num_epoch + 1,
            log_columns_test,
            "TEST",
            "ALL_TEST_RESULTS",
        )

        now = datetime.now()
        end_time = now.strftime("%Y-%m-%d_%H_%M_%S")
        print("End Time =", end_time)
        return self.model

    def test_model(self, test_data, model, loss_func, logArr, epoch):
        ### ______________ TEST ______________ ###
        with torch.no_grad():
            # Model should not improve
            model.eval()
            self.log_index_test = 0
            index = 0
            sum_losses = 0.0
            for i, z in enumerate(test_data):
                inputs = z["x"]
                y = z["y"]
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
                    model_out_val = model_out[j].item()
                    rModel_out = round(model_out_val, 4)
                    rLabel = round(y.item(), 4)
                    rLoss_val = round(loss_val.item(), 4)
                    index = index + 1
                    sum_losses = sum_losses + rLoss_val
                    avg_loss = round(sum_losses / (i + 1), 4)
                test_logging_arr = self.create_log_entry_test(
                    logArr, epoch, i, rModel_out, rLabel, rLoss_val, avg_loss
                )

        return test_logging_arr

    @staticmethod
    def training_step(model, x, y, loss, optimizer):
        """
        TODO: Needs to be filled by Aykan
        """
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

    def create_log_entry_train(
        self,
        arr,
        e,
        i,
        c,
        rModel_out,
        rLabel,
        rLoss,
        epochRunningAvgLoss,
        rRunningAvgLoss,
    ):
        """
        TODO: Needs to be filled by Aykan
        """
        arr[self.log_index_train] = [
            e,
            i,
            c,
            rModel_out,
            rLabel,
            rLoss,
            epochRunningAvgLoss,
            rRunningAvgLoss,
        ]
        self.log_index_train += 1
        return arr

    def create_log_entry_test(
        self, arr, e, i, rModel_out, rLabel, rLoss, epochRunningAvgLoss
    ):
        """
        TODO: Needs to be filled by Aykan
        """
        arr[self.log_index_test] = [
            e,
            i,
            rModel_out,
            rLabel,
            rLoss,
            epochRunningAvgLoss,
        ]
        self.log_index_test += 1
        return arr
