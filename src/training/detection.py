import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score


def smooth_and_binarize(prediction_series, window_size=5, threshold=0.5):
    """
    Smooth and binarize prediction results

    Parameters:
    -----------
    prediction_series : pd.Series
        Prediction series
    window_size : int
        Smoothing window size
    threshold : float
        Binarization threshold

    Returns:
    --------
    pd.Series
        Binarized result
    """
    smoothed = prediction_series.rolling(window=window_size, min_periods=1).mean()
    return (smoothed >= threshold).astype(int)


def detect_change_points(series, window_size=6,
                         rise_pattern=[0, 0, 1, 1, 1, 1],
                         fall_pattern=[1, 1, 0, 0, 0, 0]):
    """
    Detect change points

    Parameters:
    -----------
    series : pd.Series
        Input series
    window_size : int
        Detection window size
    rise_pattern : list
        Rising pattern
    fall_pattern : list
        Falling pattern

    Returns:
    --------
    pd.Series
        Change point detection results
    """
    cp = np.zeros(len(series), dtype=int)

    if len(series) < window_size:
        return pd.Series(cp, index=series.index)

    windows = np.lib.stride_tricks.sliding_window_view(series, window_size)
    is_rise = np.all(windows == rise_pattern, axis=1)
    is_fall = np.all(windows == fall_pattern, axis=1)

    matches = is_rise | is_fall
    cp[window_size - 1:] = matches

    return pd.Series(cp, index=series.index)


def normalized_cosine_similarity(preds, targets):
    """
    Normalized cosine similarity

    Parameters:
    -----------
    preds : torch.Tensor
        Predicted values
    targets : torch.Tensor
        Ground truth values

    Returns:
    --------
    torch.Tensor
        Similarity scores
    """
    cos_sim = F.cosine_similarity(preds, targets, dim=1)
    return (cos_sim + 1.0) / 2.0


def adjust_threshold(model, val_loader, val_anomalys):
    """
    Adjust optimal threshold

    Parameters:
    -----------
    model : nn.Module
        Model
    val_loader : DataLoader
        Validation data loader
    val_anomalys : np.ndarray
        Validation anomaly labels

    Returns:
    --------
    tuple
        Best threshold, best F1, mean, std
    """
    model.eval()
    sims = []

    with torch.no_grad():
        for x, y in val_loader:
            preds = model(x)
            sim = normalized_cosine_similarity(preds, y).cpu().numpy().flatten()
            sims.extend(sim.tolist())

    # Search for best threshold
    thresholds = np.linspace(np.percentile(sims, 5), np.percentile(sims, 95), 100)
    mean = np.mean(sims)
    std = np.std(sims)

    best_f1 = 0
    best_thresh = 0

    for thresh in thresholds:
        preds = (sims < thresh).astype(int)
        current_f1 = f1_score(val_anomalys, preds, zero_division=0)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_thresh = thresh

    model.threshold = best_thresh
    return best_thresh, best_f1, mean, std


def detect_anomalies(model, test_loader, median, std):
    """
    Detect anomalies

    Parameters:
    -----------
    model : nn.Module
        Model
    test_loader : DataLoader
        Test data loader
    median : float
        Similarity median
    std : float
        Similarity standard deviation

    Returns:
    --------
    np.ndarray
        Anomaly detection results
    """
    model.eval()
    anomalies = []

    with torch.no_grad():
        for x_test, y_test in test_loader:
            pred_test = model(x_test)

            # Calculate similarity
            similarities = normalized_cosine_similarity(
                pred_test.flatten(start_dim=1),
                y_test.flatten(start_dim=1)
            ).cpu().numpy()

            # Anomaly judgment
            lower_bound = median + model.threshold * std
            anomaly_mask = similarities < lower_bound
            anomalies.append(anomaly_mask)

    return np.concatenate(anomalies, axis=0).astype(int)


def calculate_sim_median_std(model, val_loader):
    """
    Calculate mean and std of validation set similarities

    Parameters:
    -----------
    model : nn.Module
        Model
    val_loader : DataLoader
        Validation data loader

    Returns:
    --------
    tuple
        Mean, std
    """
    model.eval()
    sims = []

    with torch.no_grad():
        for x, y in val_loader:
            preds = model(x)
            sim = normalized_cosine_similarity(preds, y).cpu().numpy().flatten()
            sims.extend(sim.tolist())

    return np.mean(sims), np.std(sims)