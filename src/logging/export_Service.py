import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt


class ExportService:
    '''
        param:
            @folderPath: base path to export to, checks if already exist otherwise create folder
    '''

    def __init__(self, folderPath):
        self.folderPath = folderPath
        if not os.path.exists(self.folderPath):
            os.makedirs(self.folderPath)

    def saveAllResults(self, arr: np.ndarray, columns: list, subFolderPath: str, pathToSave: str):
        df = pd.DataFrame(arr, columns=columns)
        folderPath = os.path.join(self.folderPath, subFolderPath)
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        filePath = os.path.join(folderPath, pathToSave + '.csv')
        df.to_csv(filePath, sep=';', index=False)

    def createLossPlot(self, arr, LOSS_PLOT_INTERVAL, logColumns_train, subFolder, pathToSave):
        folderPath = os.path.join(self.folderPath, subFolder)
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        filePath = os.path.join(folderPath, pathToSave)
        # ["EPOCH", "idx", "rIDX", "prediction", "label", "LOSS", "epochAvgLOSS", "rAvgLOSS"]
        df = pd.DataFrame(arr, columns=logColumns_train)
        # RUNNING AVG, exlude extrema
        running_avg_arr = df['rAvgLOSS'].to_numpy()
        running_avg_arr = running_avg_arr[running_avg_arr < np.percentile(running_avg_arr, 95)]
        y_min = np.min(running_avg_arr)
        y_max = np.max(running_avg_arr) * 1.3
        x_min = 0
        x_max = len(running_avg_arr)
        # Set the limits for the plot
        plt.figure(figsize=(20, 5))
        plt.plot(range(x_max), running_avg_arr, linewidth=1)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title("running_avg loss", color='red')
        plt.savefig(filePath + 'running_avg' + '.png')
        plt.clf()
        ##########################
        ######LOSS Absolute#######
        abs_loss_arr = df['LOSS'].to_numpy()
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
        plt.title("Loss Absolute", color='red')
        plt.savefig(filePath + 'absolute_loss' + '.png')
        plt.clf()
        ##########################
        ######LOSS LOSS_PLOT_INTERVAL#######
        df = df.iloc[::LOSS_PLOT_INTERVAL, :]
        df.plot(y=['prediction', 'label'], kind="line", figsize=(20, 5), lw=1, alpha=0.4)
        plt.title("soll_ist Vergleich", color='red')
        plt.savefig(filePath + 'soll_ist' + '.png')
        plt.clf()
        #

    def createLogAndPlot_epoch_test(self, arr, e, columns, subFolder, pathToSave):
        folderPath = os.path.join(self.folderPath, subFolder, '_' + str(e))
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        filePath = os.path.join(folderPath, pathToSave)
        # ["EPOCH", "idx", "prediction", "label", "LOSS", "epochAvgLOSS"]
        df = pd.DataFrame(arr, columns=columns)
        df.to_csv(filePath + '_' + str(e) + '_' + '.csv', sep=';', index=False)

        # RUNNING AVG, exlude extrema
        loss_arr = df['LOSS'].to_numpy()
        avg_loss_arr = df['epochAvgLOSS'].to_numpy()
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
        plt.title("test epoch loss", color='red')
        plt.savefig(filePath + 'epoch_test_loss' + '_' + str(e) + '.png')
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
        plt.title("test epoch avg_loss", color='red')
        plt.savefig(filePath + 'epoch_test_avgLoss' + '_' + str(e) + '.png')
        plt.clf()

        #SOLL_IST vergleich#
        df = df.iloc[::1000, :]
        df.plot(y=['prediction', 'label'], kind="line", figsize=(40, 10), lw=0.8, alpha=0.9)
        plt.title("soll_ist Vergleich", color='red')
        plt.savefig(filePath + 'soll_ist' + '.png')
        plt.clf()

    def storeGAFNumpyFeatureDataSeperatly(self, arr, FEATURE_LIST, pathToSave):
        """
            param:
                arr = dimension (1200, 20) = (länger aller ts, länger einzelner_ts)

            speichert anz_features oft (len_allSeries, singleSeries, singleSeries) = 1000x20x20 das 9 mal gespeichert
        """
        i = 0
        for fArr in arr:
            self.storeNumpyTimeSeries(fArr, FEATURE_LIST[i] + '_' + pathToSave)
            i = i + 1
        print('All saved')

    def storeNumpyTimeSeries(self, arr, pathToSave):
        """
            #tod rename into "storeSeries"
            von storeTimeSeriesNumpyFeatureDataSeperatly
            wird x mal aufgerufen x=feature List

            param:
                arr = dimension (1200, 20) = (länger aller ts, länger einzelner_ts)
        """
        filePath = os.path.join(self.folderPath, pathToSave)
        np.save(filePath, arr)

    def storeModel(self, model, modelName: str):
        model_scripted = torch.jit.script(model)  # Export to TorchScript
        now = datetime.now()
        dateTimeAsStr = str(now.day) + str(now.month) + str(now.year) + "_" + str(now.hour) + str(now.minute)
        model_scripted.save(os.path.join(self.folderPath, modelName + dateTimeAsStr + '.pt'))  # Save
