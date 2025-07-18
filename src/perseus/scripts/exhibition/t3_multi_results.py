# --- Add MultiGAT/MultiGraphSAGE results from multi_run.py ---

import pickle
import numpy as np
import pandas as pd
import torch
from os import path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_curve,
    auc,
)
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from perseus.settings import PROJECT_ROOT
from perseus.dataset.dataset_preparation import (
    get_split_data_pickle_wot,
    get_split_data_pickle_btc,
)


def extract_data_from_loader(data_loader):
    """
    Extract features and labels from a PyTorch DataLoader.
    """
    X_list, y_list = [], []
    for batch in data_loader:
        X_list.append(batch.x)
        y_list.append(batch.y)
    X = torch.cat(X_list, dim=0).numpy()
    y = torch.cat(y_list, dim=0).numpy().flatten()
    return X, y


def find_best_f1_threshold(probs, labels):
    """
    Find the threshold that results in the highest F1 score.
    """
    thresholds = np.arange(0.01, 1.00, 0.01)
    best_threshold, max_f1 = 0.01, 0
    for threshold in thresholds:
        preds = (probs > threshold).astype(int)
        f1 = f1_score(labels, preds, zero_division="warn")
        if f1 > max_f1:
            max_f1 = f1
            best_threshold = threshold
    return best_threshold


def calculate_metrics_at_threshold(probs, labels, threshold=0.54):
    """
    Calculate accuracy, precision, recall, F1 score, and MCC for given probabilities at a specified threshold.
    """
    preds = (probs > threshold).astype(int)
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division="warn")
    recall = recall_score(labels, preds, zero_division="warn")
    f1 = f1_score(labels, preds, zero_division="warn")
    mcc = matthews_corrcoef(labels, preds)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
    }


def get_metrics(results, datasets, models):
    """
    Get accuracy, precision, recall, and f1 for each model and dataset at the best F1 threshold.
    """
    metrics = {}
    for dataset in datasets:
        for model in models:
            probs = results[dataset][model]["metrics"]["probs"].flatten()
            labels = results[dataset][model]["metrics"]["labels"].flatten()
            threshold = find_best_f1_threshold(probs, labels)
            metrics_result = calculate_metrics_at_threshold(
                probs, labels, threshold=threshold
            )
            metrics[(dataset, model)] = metrics_result
    return metrics


def get_dataframe(metrics):
    """
    Convert metrics dictionary to a pandas DataFrame.
    """
    df = pd.DataFrame(metrics).T
    df.index = pd.MultiIndex.from_tuples(df.index, names=["dataset", "model"])
    df = df.reset_index()
    return df


def train_and_evaluate_rf(train_loader, test_loader, model_dataset_name):
    """
    Train and evaluate a Random Forest model, returning a result dict for DataFrame.
    """
    X_train, y_train = extract_data_from_loader(train_loader)
    X_test, y_test = extract_data_from_loader(test_loader)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    # Ensure y_scores_rf is a 1D numpy array for thresholding
    y_scores_rf = np.asarray(rf_model.predict_proba(X_test))[:, 1]
    thresholds = np.linspace(0.01, 0.99, 100)
    f1_scores = [
        f1_score(y_test, y_scores_rf > t, zero_division="warn") for t in thresholds
    ]
    best_threshold = thresholds[np.argmax(f1_scores)]
    y_pred_rf_optimal = (y_scores_rf >= best_threshold).astype(int)
    mcc = matthews_corrcoef(y_test, y_pred_rf_optimal)
    precision = precision_score(y_test, y_pred_rf_optimal, zero_division="warn")
    recall = recall_score(y_test, y_pred_rf_optimal, zero_division="warn")
    f1 = f1_score(y_test, y_pred_rf_optimal, zero_division="warn")
    accuracy = accuracy_score(y_test, y_pred_rf_optimal)
    return {
        "model_dataset": model_dataset_name,
        "precision": precision,
        "f1": f1,
        "accuracy": accuracy,
        "recall": recall,
        "mcc": mcc,
    }


if __name__ == "__main__":
    datasets = ["DDINA", "DDM"]
    models = ["GCN", "GAT", "GraphSAGE"]

    # Load results
    with open(path.join(PROJECT_ROOT, "data", "results_comparison.pkl"), "rb") as file:
        results_tr = pickle.load(file)

    # Get metrics and DataFrame
    three_features = get_metrics(results_tr, datasets, models)
    three_features_df = get_dataframe(three_features)
    three_features_df["model_dataset"] = (
        three_features_df["model"] + " - " + three_features_df["dataset"]
    )
    three_features_df = three_features_df[
        ["model_dataset", "accuracy", "precision", "recall", "f1", "mcc"]
    ]

    # Prepare special rows for Cossine similarity GCN
    a = three_features_df.iloc[0].copy()
    a["model_dataset"] = "Cossine similarity GCN (directed features)"
    a_df = a.to_frame().T
    b = three_features_df.iloc[3].copy()
    b["model_dataset"] = "Cossine similarity GCN (weighted features)"
    b_df = b.to_frame().T

    # Filter and merge DataFrames
    filtered_df = three_features_df[
        ~three_features_df["model_dataset"].str.contains("GCN|COSS")
    ]
    merged_df = pd.concat([filtered_df, a_df, b_df], ignore_index=True)
    merged_df["model_dataset"] = merged_df["model_dataset"].replace(
        {
            "GAT - DDINA": "Directed diffusion GAT",
            "GraphSAGE - DDINA": "Directed diffusion GraphSAGE",
            "GAT - DDM": "Weighted diffusion GAT",
            "GraphSAGE - DDM": "Weighted diffusion GraphSAGE",
        }
    )
    df_reordered = merged_df[
        ["model_dataset", "precision", "f1", "accuracy", "recall", "mcc"]
    ]
    df_reordered = df_reordered.loc[[3, 1, 0, 2, 4, 5]].reset_index(drop=True)

    # Random Forest results
    rf_rows = []
    # Weighted Diffusion without Topological Features
    train_loader, _, test_loader = get_split_data_pickle_wot("DDM")
    rf_rows.append(
        train_and_evaluate_rf(
            train_loader,
            test_loader,
            "Random Forest Weighted Diffusion without Topological Features",
        )
    )
    # Weighted Diffusion
    train_loader, _, test_loader = get_split_data_pickle_btc("DDM")
    rf_rows.append(
        train_and_evaluate_rf(
            train_loader, test_loader, "Random Forest Weighted Diffusion"
        )
    )
    # Directed Diffusion
    train_loader_d, _, test_loader_d = get_split_data_pickle_btc("DDINA")
    rf_rows.append(
        train_and_evaluate_rf(
            train_loader_d, test_loader_d, "Random Forest Directed Diffusion"
        )
    )
    for row in rf_rows:
        df_reordered = df_reordered.append(row, ignore_index=True)

    # Fusion/multi results
    with open(
        path.join(PROJECT_ROOT, "data/new_results_btc_batched.pkl"),
        "rb",
    ) as f:
        fusion_results = pickle.load(f)
    batch_size = 8
    fusion_rows = []
    model_map = {"MultiGAT": "Fusion GAT", "MultiGraphSAGE": "Fusion GraphSAGE"}
    dataset_map = {"DDINA": "Directed diffusion", "DDM": "Weighted diffusion"}
    for data in ["DDINA", "DDM"]:
        for model in ["MultiGAT", "MultiGraphSAGE"]:
            metrics = fusion_results[data][model][batch_size]["metrics"]
            labels = metrics["labels"].flatten()
            probs = metrics["probs"].flatten()
            threshold = find_best_f1_threshold(probs, labels)
            preds = (probs > threshold).astype(int)
            precision = precision_score(labels, preds, zero_division="warn")
            f1 = f1_score(labels, preds, zero_division="warn")
            accuracy = accuracy_score(labels, preds)
            recall = recall_score(labels, preds, zero_division="warn")
            mcc = matthews_corrcoef(labels, preds)
            fusion_rows.append(
                {
                    "model_dataset": f"{model_map[model]} {dataset_map[data]}",
                    "precision": precision,
                    "f1": f1,
                    "accuracy": accuracy,
                    "recall": recall,
                    "mcc": mcc,
                }
            )
    fusion_df = pd.DataFrame(fusion_rows)
    df_reordered = pd.concat([df_reordered, fusion_df], ignore_index=True)
    # SOTA renaming
    df_reordered.loc[4, "model_dataset"] = "SOTA Directed Diffusion"
    df_reordered.loc[5, "model_dataset"] = "SOTA Weighted Diffusion"
    df_reordered["model_dataset"] = df_reordered["model_dataset"].str.replace(
        "Multi-head", "Fusion"
    )
