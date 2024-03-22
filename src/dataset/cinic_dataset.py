import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class CinicDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe

    def __getitem__(self, idx: int):
        """
        Returns an observation from a given index from the dataset.
        """
        return {
            "image": np.array(eval(self.df.iloc[idx]["image"])),
            "label": self.df.iloc[idx]["label"],
        }

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.df)
