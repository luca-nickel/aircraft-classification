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
        folder_path = os.path.join(self.folder_path, sub_folder_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_path = os.path.join(folder_path, path_to_save + ".csv")
        df.to_csv(file_path, sep=";", index=False)

    def store_numpy_time_series(self, array, path_to_save):
        file_path = os.path.join(self.folder_path, path_to_save)
        np.save(file_path, array)

    def store_model(self, model, model_name: str):
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
            os.path.join(self.folder_path, model_name + date_time_as_str + ".pt")
        )  # Save
