import os
from datetime import datetime

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
        df = pd.DataFrame(arr, columns=columns)
        folderPath = os.path.join(self.folder_path, sub_folder_path)
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        filePath = os.path.join(folderPath, path_to_save + ".csv")
        df.to_csv(filePath, sep=";", index=False)

    def store_numpy_time_series(self, arr, pathToSave):
        """
        #tod rename into "storeSeries"
        von storeTimeSeriesNumpyFeatureDataSeperatly
        wird x mal aufgerufen x=feature List

        param:
            arr = dimension (1200, 20) = (länger aller ts, länger einzelner_ts)
        """
        filePath = os.path.join(self.folder_path, pathToSave)
        np.save(filePath, arr)

    def store_model(self, model, modelName: str):
        model_scripted = torch.jit.script(model)  # Export to TorchScript
        now = datetime.now()
        date_time_as_str = (
            str(now.day)
            + str(now.month)
            + str(now.year)
            + "_"
            + str(now.hour)
            + str(now.minute)
        )
        model_scripted.save(
            os.path.join(self.folder_path, modelName + date_time_as_str + ".pt")
        )  # Save
