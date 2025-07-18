"""
Module for running fusion experiments with GNN models on graph datasets.
Provides training, evaluation, and experiment orchestration utilities for MultiGAT and MultiGraphSAGE models.
"""

import itertools
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.loader import DataListLoader
from perseus.dataset.dataset_preparation import get_split_data_pickle_btc_noloader
from perseus.model.gnn_models import (
    MultiGAT,
    MultiGraphSAGE,
)
from concurrent.futures import ProcessPoolExecutor, as_completed
from perseus.settings import PROJECT_ROOT
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    matthews_corrcoef,
)
import pickle
from sklearn.metrics import roc_curve
import os
from torch.optim.adam import Adam
import concurrent.futures
import random
import numpy as np


def set_seed(seed=7):
    """
    Sets random seed for reproducibility.
    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    Args:
        model (torch.nn.Module): The model to train.
        loader (DataListLoader): Data loader for training data.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        criterion: Loss function.
        device (torch.device): Device to run computations on.
    Returns:
        tuple: (average loss, accuracy) for the epoch.
    """
    model.train()
    total_loss = total_correct = total_nodes = 0
    for data_list in loader:
        data_list = [d.to(device) for d in data_list]
        logits_list = model(data_list)
        y_list = [d.y.view(-1).float() for d in data_list]
        logits = torch.cat(logits_list, dim=0)
        y = torch.cat(y_list, dim=0)
        optimizer.zero_grad()
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        preds = (logits.sigmoid() > 0.5).long()
        total_correct += (preds == y.long()).sum().item()
        total_nodes += y.size(0)
    return total_loss / total_nodes, total_correct / total_nodes


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    """
    Evaluates the model for one epoch.
    Args:
        model (torch.nn.Module): The model to evaluate.
        loader (DataListLoader): Data loader for evaluation data.
        criterion: Loss function.
        device (torch.device): Device to run computations on.
    Returns:
        tuple: (average loss, accuracy, f1, precision, recall, mcc) for the epoch.
    """
    model.eval()
    total_loss = total_correct = total_nodes = 0
    all_preds, all_y = [], []
    for data_list in loader:
        data_list = [d.to(device) for d in data_list]
        logits_list = model(data_list)
        y_list = [d.y.view(-1).float() for d in data_list]
        logits = torch.cat(logits_list, dim=0)
        y = torch.cat(y_list, dim=0)
        loss = criterion(logits, y)
        total_loss += loss.item() * y.size(0)
        preds = (logits.sigmoid() > 0.5).long()
        total_correct += (preds == y.long()).sum().item()
        total_nodes += y.size(0)
        all_preds.append(preds.cpu())
        all_y.append(y.cpu())
    all_preds = torch.cat(all_preds).numpy()
    all_y = torch.cat(all_y).numpy()
    f1 = f1_score(all_y, all_preds, zero_division=0)
    precision = precision_score(all_y, all_preds, zero_division=0)
    recall = recall_score(all_y, all_preds, zero_division=0)
    acc = accuracy_score(all_y, all_preds)
    mcc = matthews_corrcoef(all_y, all_preds)
    return (total_loss / total_nodes, acc, f1, precision, recall, mcc)


def run_experiment(
    data_name,
    model_cls,
    hidden_channels,
    lr,
    weight_decay,
    epochs=100,
    batch_size=8,
    device_str=None,
    return_roc=False,
    return_timings=False,
    return_embs=False,
    seed=42,
):
    """
    Runs a single experiment with the specified model and hyperparameters.
    Args:
        data_name (str): Name of the dataset.
        model_cls (type): Model class to instantiate.
        hidden_channels (int): Number of hidden channels in the model.
        lr (float): Learning rate.
        weight_decay (float): Weight decay for optimizer.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for data loader.
        device_str (str, optional): Device string (e.g., 'cuda').
        return_roc (bool): Whether to return ROC curve data.
        return_timings (bool): Whether to return timing information.
        return_embs (bool): Whether to return embeddings.
        seed (int): Random seed for reproducibility.
    Returns:
        dict: Results including metrics, model, and optionally ROC/timing/embedding data.
    """
    set_seed(seed)
    import time
    from sklearn.metrics import roc_curve

    device = (
        torch.device(device_str)
        if device_str
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    train_list, test_list, valid_list = get_split_data_pickle_btc_noloader(data_name)
    train_loader = DataListLoader(train_list, batch_size=batch_size, shuffle=True)
    valid_loader = DataListLoader(valid_list, batch_size=batch_size)
    test_loader = DataListLoader(test_list, batch_size=batch_size)
    in_channels = train_list[0].x.size(1)
    model = model_cls(in_channels=in_channels, hidden_channels=hidden_channels).to(
        device
    )
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_val_f1 = 0.0
    best_state = None
    train_times = []
    for epoch in range(1, epochs + 1):
        if return_timings:
            start_time = time.time()
        train_epoch(model, train_loader, optimizer, criterion, device)
        if return_timings:
            train_times.append(time.time() - start_time)
        _, _, val_f1, _, _, _ = eval_epoch(model, valid_loader, criterion, device)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict()
    model.load_state_dict(best_state)
    # Collect all_probs and all_labels on test set
    model.eval()
    all_probs, all_labels = [], []
    batch_times, num_nodes, embs = [], [], []
    with torch.no_grad():
        for data_list in test_loader:
            data_list = [d.to(device) for d in data_list]
            if return_timings:
                batch_start = time.time()
            logits_list = model(data_list)
            if return_timings:
                batch_times.append(time.time() - batch_start)
            y_list = [d.y.view(-1).float() for d in data_list]
            x_list = [d.x for d in data_list]
            logits = torch.cat(logits_list, dim=0)
            y = torch.cat(y_list, dim=0)
            all_probs.append(logits.cpu())
            all_labels.append(y.cpu())
            if return_embs:
                # Optionally collect embeddings if model supports it
                if hasattr(model, "get_embeddings"):
                    embs.append(model.get_embeddings(data_list))
                else:
                    embs.append([])
            for x in x_list:
                num_nodes.append(x.shape[0])
    all_probs = torch.cat(all_probs, dim=0).sigmoid().numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    _, test_acc, test_f1, test_precision, test_recall, test_mcc = eval_epoch(
        model, test_loader, criterion, device
    )
    results = {
        "data": data_name,
        "model": model_cls.__name__,
        "hidden_channels": hidden_channels,
        "lr": lr,
        "weight_decay": weight_decay,
        "best_val_f1": best_val_f1,
        "test_acc": test_acc,
        # "test_loss": test_loss,
        "test_f1": test_f1,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_mcc": test_mcc,
        "model_instance": model,
        "model_weights": model.state_dict(),
        "all_probs": all_probs,
        "all_labels": all_labels,
        "batch_size": batch_size,
    }
    if return_roc:
        if len(all_labels.shape) == 1 or (
            hasattr(all_labels, "shape") and all_labels.shape[1] == 1
        ):
            fpr, tpr, _ = roc_curve(all_labels.ravel(), all_probs.ravel())
            results["fpr"] = {0: fpr}
            results["tpr"] = {0: tpr}
        else:
            fpr_dict, tpr_dict = {}, {}
            for i in range(all_labels.shape[1]):
                fpr, tpr, _ = roc_curve(all_labels[:, i], all_probs[:, i])
                fpr_dict[i] = fpr
                tpr_dict[i] = tpr
            results["fpr"] = fpr_dict
            results["tpr"] = tpr_dict
    if return_timings:
        results["train_times"] = train_times
        results["batch_times"] = batch_times
        results["num_nodes"] = num_nodes
    if return_embs:
        results["embs"] = embs
    return results


def collect_full_outputs(
    data_name,
    model_cls,
    hidden_channels,
    lr,
    weight_decay,
    batch_sizes=[8],
    epochs=100,
    device_str=None,
):
    """
    Runs experiments for multiple batch sizes and collects full outputs.
    Args:
        data_name (str): Name of the dataset.
        model_cls (type): Model class to instantiate.
        hidden_channels (int): Number of hidden channels.
        lr (float): Learning rate.
        weight_decay (float): Weight decay for optimizer.
        batch_sizes (list of int): List of batch sizes to try.
        epochs (int): Number of training epochs.
        device_str (str, optional): Device string.
    Returns:
        dict: Results for each batch size.
    """
    results_by_batch_size = {}
    for batch_size in batch_sizes:
        result = run_experiment(
            data_name,
            model_cls,
            hidden_channels,
            lr,
            weight_decay,
            epochs,
            batch_size,
            device_str,
            return_roc=True,
            return_timings=True,
            return_embs=True,
        )
        labels = result["all_labels"]
        probs = result["all_probs"]
        results_by_batch_size[batch_size] = {
            "metrics": {
                "labels": labels,
                "probs": probs,
            },
            "model": result["model_instance"],
            "model_weights": result["model_weights"],
            "batch_size": batch_size,
            "fpr": result.get("fpr", {}),
            "best_val_f1": result.get("best_val_f1", 0),
            "tpr": result.get("tpr", {}),
            "train_times": result.get("train_times", []),
            "batch_times": result.get("batch_times", []),
            "num_nodes": result.get("num_nodes", []),
            "embs": result.get("embs", []),
            "lr": lr,
            "test_acc": result.get("test_acc", 0),
            "test_f1": result.get("test_f1", 0),
            "test_precision": result.get("test_precision", 0),
            "test_recall": result.get("test_recall", 0),
            "test_mcc": result.get("test_mcc", 0),
            "hidden_channels": hidden_channels,
            "weight_decay": weight_decay,
        }

    return results_by_batch_size


def parameter_search(
    data_names=["DDM", "DDINA"],
    model_names=["MultiGAT", "MultiGraphSAGE"],
    hidden_channels_list=[8, 32, 64, 128, 512],
    lr_list=[0.05, 0.005, 0.0005, 0.00005, 0.000005],
    weight_decay_list=[5e-4],
    batch_sizes=range(8, 9),
    # out_path=os.path.join(PROJECT_ROOT, "data",  "new_results_btc_711.pkl"),
    param_tuning_out_path=os.path.join(
        PROJECT_ROOT, "data", "param_tuning_results.pkl"
    ),
):
    """
    Runs experiments for all combinations of parameters and exports results for plotting.
    Args:
        data_names (list of str): List of dataset names.
        model_names (list of str): List of model names.
        hidden_channels_list (list of int): List of hidden channel sizes.
        lr_list (list of float): List of learning rates.
        weight_decay_list (list of float): List of weight decays.
        batch_sizes (iterable of int): Batch sizes to try.
        out_path (str): Output path for results pickle file.
        param_tuning_out_path (str): Output path for param tuning results pickle file.
    Returns:
        None. Saves results to files.
    """

    model_map = {"MultiGAT": MultiGAT, "MultiGraphSAGE": MultiGraphSAGE}
    results = {d: {} for d in data_names}
    param_tuning_results = []
    param_grid = list(
        itertools.product(
            data_names, model_names, hidden_channels_list, lr_list, weight_decay_list
        )
    )
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        future_to_cfg = {}
        for data_name, model_name, hidden_channels, lr, weight_decay in param_grid:
            model_cls = model_map[model_name]
            print(
                f"Submitting full output for {data_name} {model_name} (hidden={hidden_channels}, lr={lr}, wd={weight_decay})..."
            )
            future = executor.submit(
                collect_full_outputs,
                data_name,
                model_cls,
                hidden_channels,
                lr,
                weight_decay,
                list(batch_sizes),
                100,
            )
            future_to_cfg[future] = (
                data_name,
                model_name,
                hidden_channels,
                lr,
                weight_decay,
            )
        for future in concurrent.futures.as_completed(future_to_cfg):
            data_name, model_name, hidden_channels, lr, weight_decay = future_to_cfg[
                future
            ]
            try:
                batch_size_results = future.result()
                results[data_name][model_name] = batch_size_results
                # Collect param tuning results for each batch size
                for bsz, info in batch_size_results.items():
                    param_tuning_results.append(
                        {
                            "batch_size": bsz,
                            "lr": info.get("lr", lr),
                            "hidden_channels": info.get(
                                "hidden_channels", hidden_channels
                            ),
                            "model": model_name,
                            "data": data_name,
                            "best_val_f1": info.get("best_val_f1", 0),
                        }
                    )
                print(f"Completed: {data_name} {model_name}")
            except Exception as exc:
                print(f"Error with {data_name} {model_name}: {exc}")
    # with open(out_path, "wb") as f:
    #     pickle.dump(results, f)
    # print(f"Saved results for plotting to {out_path}")
    # Save param tuning results as flat list
    with open(param_tuning_out_path, "wb") as f:
        pickle.dump(param_tuning_results, f)
    print(f"Saved param tuning results to {param_tuning_out_path}")


def batched_experiment(
    data_names=["DDM", "DDINA"],
    model_names=["MultiGAT", "MultiGraphSAGE"],
    hidden_channels_list=[32],
    lr_list=[0.0005],
    weight_decay_list=[5e-4],
    batch_sizes=range(2, 21, 2),
    out_path=os.path.join(PROJECT_ROOT, "data", "new_results_btc_batched.pkl"),
    # param_tuning_out_path=os.path.join(
    #     PROJECT_ROOT, "data", "param_tuning_results.pkl"
    # ),
):
    """
    Runs experiments for all combinations of parameters and exports results for plotting.
    Args:
        data_names (list of str): List of dataset names.
        model_names (list of str): List of model names.
        hidden_channels_list (list of int): List of hidden channel sizes.
        lr_list (list of float): List of learning rates.
        weight_decay_list (list of float): List of weight decays.
        batch_sizes (iterable of int): Batch sizes to try.
        out_path (str): Output path for results pickle file.
        param_tuning_out_path (str): Output path for param tuning results pickle file.
    Returns:
        None. Saves results to files.
    """

    model_map = {"MultiGAT": MultiGAT, "MultiGraphSAGE": MultiGraphSAGE}
    results = {d: {} for d in data_names}
    param_tuning_results = []
    param_grid = list(
        itertools.product(
            data_names, model_names, hidden_channels_list, lr_list, weight_decay_list
        )
    )
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        future_to_cfg = {}
        for data_name, model_name, hidden_channels, lr, weight_decay in param_grid:
            model_cls = model_map[model_name]
            print(
                f"Submitting full output for {data_name} {model_name} (hidden={hidden_channels}, lr={lr}, wd={weight_decay})..."
            )
            future = executor.submit(
                collect_full_outputs,
                data_name,
                model_cls,
                hidden_channels,
                lr,
                weight_decay,
                list(batch_sizes),
                100,
            )
            future_to_cfg[future] = (
                data_name,
                model_name,
                hidden_channels,
                lr,
                weight_decay,
            )
        for future in concurrent.futures.as_completed(future_to_cfg):
            data_name, model_name, hidden_channels, lr, weight_decay = future_to_cfg[
                future
            ]
            try:
                batch_size_results = future.result()
                results[data_name][model_name] = batch_size_results
                # Collect param tuning results for each batch size
                for bsz, info in batch_size_results.items():
                    param_tuning_results.append(
                        {
                            "batch_size": bsz,
                            "lr": info.get("lr", lr),
                            "hidden_channels": info.get(
                                "hidden_channels", hidden_channels
                            ),
                            "model": model_name,
                            "data": data_name,
                            "best_val_f1": info.get("best_val_f1", 0),
                        }
                    )
                print(f"Completed: {data_name} {model_name}")
            except Exception as exc:
                print(f"Error with {data_name} {model_name}: {exc}")
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved results for plotting to {out_path}")
    # Save param tuning results as flat list
    # with open(param_tuning_out_path, "wb") as f:
    #     pickle.dump(param_tuning_results, f)
    # print(f"Saved param tuning results to {param_tuning_out_path}")


if __name__ == "__main__":
    parameter_search()
    batched_experiment()
    # Export results for plotting
    # export_results_for_plot()
