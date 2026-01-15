import torch
from torch.utils.data import Dataset
import numpy as np


class TSData(Dataset):
    """Time series dataset"""

    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def create_dataset(df, time_steps=10, feature_cols=None,
                   label_col=None, change_col=None):
    """
    Create time series dataset

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    time_steps : int
        Time window size
    feature_cols : list
        Feature column names
    label_col : str
        Anomaly label column name
    change_col : str
        Change point label column name

    Returns:
    --------
    tuple
        X, y, anomalies, changes
    """
    if feature_cols is None:
        raise ValueError("Feature columns must be provided as a list.")
    if label_col is None:
        raise ValueError("Label column must be provided.")
    if change_col is None:
        raise ValueError("Change column must be provided.")

    features = df[feature_cols]
    labels = df[label_col]
    changes = df[change_col]

    # Standardization
    features = (features - features.mean()) / features.std()

    X, y, a, c = [], [], [], []
    data = features.values

    for i in range(len(df) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
        a.append(labels.values[i + time_steps])
        c.append(changes.values[i + time_steps])

    return np.array(X), np.array(y), np.array(a), np.array(c)