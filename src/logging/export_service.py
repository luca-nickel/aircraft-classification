import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


class ExportService:
    """
    param:
        @folderPath: base path to export to, checks if already exist otherwise create folder
    """

    def __init__(self, folder_path):
        self.folder_path = folder_path
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

    def save_all_results(
        self, arr: np.ndarray, columns: list, sub_folder_path: str, path_to_save: str
    ):
        """
        TODO: Needs to be filled by Aykan
        """
        df = pd.DataFrame(arr, columns=columns)
        folder_path = os.path.join(self.folder_path, sub_folder_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_path = os.path.join(folder_path, path_to_save + ".csv")
        df.to_csv(file_path, sep=";", index=False)

    def create_loss_plot(
        self, arr, loss_plot_interval, log_columns_train, sub_folder, path_to_save
    ):
        """
        TODO: Needs to be filled by Aykan
        """
        folder_path = os.path.join(self.folder_path, sub_folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_path = os.path.join(folder_path, path_to_save)
        # ["EPOCH", "idx", "rIDX", "prediction", "label", "LOSS", "epochAvgLOSS", "rAvgLOSS"]
        df = pd.DataFrame(arr, columns=log_columns_train)
        # RUNNING AVG, exlude extrema
        running_avg_arr = df["rAvgLOSS"].to_numpy()
        running_avg_arr = running_avg_arr[
            running_avg_arr < np.percentile(running_avg_arr, 95)
        ]
        y_min = np.min(running_avg_arr)
        y_max = np.max(running_avg_arr) * 1.3
        x_min = 0
        x_max = len(running_avg_arr)
        # Set the limits for the plot
        plt.figure(figsize=(20, 5))
        plt.plot(range(x_max), running_avg_arr, linewidth=1)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title("running_avg loss", color="red")
        plt.savefig(file_path + "running_avg" + ".png")
        plt.clf()
        ##########################
        ######LOSS Absolute#######
        abs_loss_arr = df["LOSS"].to_numpy()
        abs_loss_arr = abs_loss_arr[abs_loss_arr < np.percentile(abs_loss_arr, 95)]
        y_min = np.min(abs_loss_arr)
        y_max = np.max(abs_loss_arr) * 1.3
        x_min = 0
        x_max = len(abs_loss_arr)
        # Set the limits for the plot
        plt.figure(figsize=(20, 5))
        plt.plot(range(x_max), abs_loss_arr, linewidth=1)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title("Loss Absolute", color="red")
        plt.savefig(file_path + "absolute_loss" + ".png")
        plt.clf()
        ##########################
        ######LOSS LOSS_PLOT_INTERVAL#######
        df = df.iloc[::loss_plot_interval, :]
        df.plot(
            y=["prediction", "label"], kind="line", figsize=(20, 5), lw=1, alpha=0.4
        )
        plt.title("soll_ist Vergleich", color="red")
        plt.savefig(file_path + "soll_ist" + ".png")
        plt.clf()
        #

    def create_log_and_plot_epoch_test(self, arr, e, columns, sub_folder, path_to_save):
        """
        TODO: Needs to be filled by Aykan
        """
        folder_path = os.path.join(self.folder_path, sub_folder, "_" + str(e))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_path = os.path.join(folder_path, path_to_save)
        # ["EPOCH", "idx", "prediction", "label", "LOSS", "epochAvgLOSS"]
        df = pd.DataFrame(arr, columns=columns)
        df.to_csv(file_path + "_" + str(e) + "_" + ".csv", sep=";", index=False)

        # RUNNING AVG, exlude extrema
        loss_arr = df["LOSS"].to_numpy()
        avg_loss_arr = df["epochAvgLOSS"].to_numpy()
        loss_arr = loss_arr[loss_arr < np.percentile(loss_arr, 95)]
        y_min = np.min(loss_arr)
        y_max = np.max(loss_arr) * 1.3
        x_min = 0
        x_max = len(loss_arr)
        # Set the limits for the plot
        plt.figure(figsize=(20, 5))
        plt.plot(range(x_max), loss_arr, linewidth=1)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title("test epoch loss", color="red")
        plt.savefig(file_path + "epoch_test_loss" + "_" + str(e) + ".png")
        plt.clf()
        ######Plot AVG LOSSS
        avg_loss_arr = avg_loss_arr[avg_loss_arr < np.percentile(avg_loss_arr, 95)]
        y_min = np.min(avg_loss_arr)
        y_max = np.max(avg_loss_arr) * 1.3
        x_min = 0
        x_max = len(avg_loss_arr)
        # Set the limits for the plot
        plt.figure(figsize=(20, 5))
        plt.plot(range(x_max), avg_loss_arr, linewidth=1)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title("test epoch avg_loss", color="red")
        plt.savefig(file_path + "epoch_test_avgLoss" + "_" + str(e) + ".png")
        plt.clf()

        # SOLL_IST vergleich#
        df = df.iloc[::1000, :]
        df.plot(
            y=["prediction", "label"], kind="line", figsize=(40, 10), lw=0.8, alpha=0.9
        )
        plt.title("soll_ist Vergleich", color="red")
        plt.savefig(file_path + "soll_ist" + ".png")
        plt.clf()

    def store_gaf_numpy_feature_data(self, arr, feature_list, path_to_save: str):
        """
        param:
            arr = dimension (1200, 20) = (länger aller ts, länger einzelner_ts)

        speichert anz_features oft (len_allSeries, singleSeries, singleSeries) = 1000x20x20 das 9 mal gespeichert
        """
        i = 0
        for feature_array in arr:
            self.store_numpy_time_series(
                feature_array, feature_list[i] + "_" + path_to_save
            )
            i = i + 1
        print("All saved")

    def store_numpy_time_series(self, arr, path_to_save: str):
        """
        #tod rename into "storeSeries"
        von storeTimeSeriesNumpyFeatureDataSeperatly
        wird x mal aufgerufen x=feature List

        param:
            arr = dimension (1200, 20) = (länger aller ts, länger einzelner_ts)
        """
        file_path = os.path.join(self.folder_path, path_to_save)
        np.save(file_path, arr)

    def store_model(self, model, model_name: str):
        """
        TODO: Needs to be filled by Aykan
        """
        model_scripted = torch.jit.script(model)  # Export to TorchScript
        now = datetime.now()
        date_time_string = (
            str(now.day)
            + str(now.month)
            + str(now.year)
            + "_"
            + str(now.hour)
            + str(now.minute)
        )
        model_scripted.save(
            os.path.join(self.folder_path, model_name + date_time_string + ".pt")
        )  # Save
