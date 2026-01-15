import os
import sys
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader


# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.transformer import TimeSeriesTransformer
from src.models.graph import build_relation_graph_raw_sequence
from src.dataset.dataset import TSData, create_dataset
from src.training.trainer import Trainer
from src.training.detection import detect_anomalies, detect_change_points, calculate_sim_median_std
from src.utils.evaluation import evaluating_change_point


def main():
    # Configuration
    DATA_DIR = "./data"
    CHECKPOINT_DIR = "checkpoints"
    TIME_STEPS = 30
    FEATURE_COLS = [
        'Accelerometer1RMS', 'Accelerometer2RMS', 'Current',
        'Pressure', 'Temperature', 'Thermocouple', 'Voltage',
        'Volume Flow RateRMS'
    ]
    LABEL_COL = 'anomaly'
    CHANGE_COL = 'changepoint'

    # Load data
    print("Loading data...")
    all_files = []
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith(".csv") and 'anomaly-free' not in file:
                all_files.append(os.path.join(root, file))

    list_of_df = [pd.read_csv(file, sep=';', index_col='datetime',
                              parse_dates=True) for file in all_files]

    print(f"Number of datasets: {len(list_of_df)}")
    print(f"Shape of random dataset: {list_of_df[0].shape}")

    # Statistics
    n_cp = sum([len(df[df.changepoint == 1.]) for df in list_of_df])
    n_outlier = sum([len(df[df.anomaly == 1.]) for df in list_of_df])
    print(f"Number of changepoints: {n_cp}")
    print(f"Number of outliers: {n_outlier}")

    # Process each dataset
    predicted_outlier = []
    predicted_cp = []

    for df_idx, df in enumerate(list_of_df):
        print(f"\n{'=' * 60}")
        print(f"Processing dataset {df_idx}")
        print(f"{'=' * 60}")

        try:
            # Create dataset
            X, y, a, c = create_dataset(
                df, time_steps=TIME_STEPS,
                feature_cols=FEATURE_COLS,
                label_col=LABEL_COL,
                change_col=CHANGE_COL
            )

            # Limit feature data for graph construction
            features = df[FEATURE_COLS][:400]
            edge_index, edge_attr = build_relation_graph_raw_sequence(
                features.values, threshold=0.3, topk=None,
                alpha=0.01, n_perm=500, n_jobs=8
            )

            print(f"Edge index shape: {edge_index.shape}")
            print(f"Edge attr shape: {edge_attr.shape}")

            # Data splitting
            train_data = X[:400]
            train_labels = y[:400]
            val_data = X[400:800]
            val_labels = y[400:800]
            val_anomalys = a[400:800]
            test_data = X
            test_labels = y
            test_anomalys = a
            test_changes = c

            # Create data loaders
            train_loader = DataLoader(
                TSData(train_data, train_labels),
                batch_size=32, shuffle=False
            )
            val_loader = DataLoader(
                TSData(val_data, val_labels),
                batch_size=32, shuffle=False
            )
            test_loader = DataLoader(
                TSData(test_data, test_labels),
                batch_size=32, shuffle=False
            )

            # Create model
            model = TimeSeriesTransformer(
                input_dim=train_data.shape[2],
                hidden_dim=64,
                edge_index=edge_index,
                edge_attr=edge_attr,
                window_size=train_data.shape[1]
            )

            # Train
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                checkpoint_dir=CHECKPOINT_DIR
            )

            model = trainer.train(val_anomalys)

            # Test
            mean, std = calculate_sim_median_std(model, val_loader)
            pred_anomalys = detect_anomalies(model, test_loader, mean, std)

            from sklearn.metrics import f1_score
            test_f1 = f1_score(test_anomalys, pred_anomalys, pos_label=1)
            print(f"Test F1: {test_f1:.4f}")

            # Save predictions
            pred_anomalys_padded = np.pad(
                pred_anomalys, (TIME_STEPS, 0), mode='constant'
            )
            predicted_outlier.append(pd.Series(pred_anomalys_padded, index=df.index))

            # Change point detection
            prediction_cp = detect_change_points(
                pd.Series(pred_anomalys_padded, index=df.index)
            )
            predicted_cp.append(prediction_cp)

        except Exception as e:
            print(f"Error processing dataset {df_idx}: {str(e)}")
            continue

    # Final evaluation
    print(f"\n{'=' * 60}")
    print("FINAL EVALUATION")
    print(f"{'=' * 60}")

    # Anomaly detection evaluation
    true_outlier = [df.anomaly for df in list_of_df]
    binary_results = evaluating_change_point(
        true_outlier, predicted_outlier,
        metric='binary', numenta_time='30sec'
    )

    # Change point detection evaluation
    true_cp = [df.changepoint for df in list_of_df]
    nab_results = evaluating_change_point(
        true_cp, predicted_cp,
        metric='nab', numenta_time='30sec'
    )

    delay_results = evaluating_change_point(
        true_cp, predicted_cp,
        metric='average_delay', numenta_time='30sec'
    )

    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()