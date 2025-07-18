"""
This script is used to run the experiments for the empirical study. It is used to train the models on the datasets and evaluate the performance of the models on the test set.
"""

import time
from os import path
import random
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
import numpy as np
from sklearn.metrics import (
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    balanced_accuracy_score,
)
from perseus.dataset.dataset_preparation import (
    get_split_data_pickle_btc,
)


from perseus.model.gnn_models import GCNNet, Net, GraphSAGENet
from perseus.settings import PROJECT_ROOT


# Set a seed value
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_experiment(model, train_loader, valid_loader, test_loader, num_epochs=100):
    """
    Run the experiment for the given model and dataset. Prepares for the training and testing of the model.

    Args:
        model (torch.nn.Module): The neural network model to train and evaluate.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
        valid_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
        num_epochs (int, optional): Number of training epochs. Defaults to 100.

    Returns:
        tuple: (
            model (torch.nn.Module): The trained model.
            model_weights (dict): The state dict of the trained model.
            fpr_dict (dict): False positive rates for each label.
            tpr_dict (dict): True positive rates for each label.
            thresholds_dict (dict): Thresholds for ROC for each label.
            train_times (list): Training time per epoch.
            batch_times (list): Inference time per batch in test set.
            num_nodes (list): Number of nodes per batch in test set.
            embs (list): Embeddings from the last layer for test set.
            all_labels (list): List of all labels in the test set.
            train_losses (list): Training loss per epoch.
            valid_losses (list): Validation loss per epoch.
        )
    """

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    # loss_op = torch.nn.BCEWithLogitsLoss()
    loss_op = torch.nn.BCELoss()

    train_times = []  # To store training times for each epoch
    train_losses, valid_losses = [], []

    def train_and_validate():
        model.train()
        for epoch in range(num_epochs):
            start_time = time.time()  # Start timing the epoch
            total_loss = 0
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                output, _ = model(data.x, data.edge_index)
                loss = loss_op(output, data.y.float())
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * data.num_graphs
            end_time = time.time()  # End timing the epoch
            epoch_time = end_time - start_time
            train_times.append(epoch_time)  # Store the time for this epoch
            train_losses.append(total_loss / len(train_loader.dataset))

            # Validation step
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for data in valid_loader:
                    data = data.to(device)
                    output, _ = model(data.x, data.edge_index)
                    val_loss += loss_op(output, data.y.float()).item() * data.num_graphs
            valid_losses.append(val_loss / len(valid_loader.dataset))
            print(
                f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {valid_losses[-1]:.4f}, Time: {epoch_time:.2f}s"
            )

    @torch.no_grad()
    def test(loader):
        model.eval()
        all_probs, all_labels, all_embs = [], [], []
        batch_times, num_nodes = [], []
        for data in loader:
            data = data.to(device)
            start_time = time.time()
            num_nodes.append(data.x.size(0))
            out, embs = model(
                data.x, data.edge_index
            )  # Make sure embs are the last layer embeddings
            batch_time = time.time() - start_time
            batch_times.append(batch_time)

            all_probs.append(out.cpu())
            all_labels.append(data.y.cpu())
            all_embs.append(embs.cpu())  # Collect embeddings

        return all_probs, all_labels, all_embs, batch_times, num_nodes

    train_and_validate()

    probs, labels, embs, batch_times, num_nodes = test(test_loader)
    probs = torch.cat(probs, dim=0).sigmoid().numpy()
    labels = torch.cat(labels, dim=0).numpy()

    # Compute ROC for each label and store
    fpr_dict, tpr_dict, thresholds_dict = {}, {}, {}
    for i in range(labels.shape[1]):  # Assuming labels is a 2D array: [samples, labels]
        fpr, tpr, thresholds = roc_curve(labels[:, i], probs[:, i])
        fpr_dict[i], tpr_dict[i], thresholds_dict[i] = fpr, tpr, thresholds

    all_labels = [torch.cat(test(test_loader)[1])]

    model_weights = model.state_dict()

    return (
        model,
        model_weights,
        fpr_dict,
        tpr_dict,
        thresholds_dict,
        train_times,
        batch_times,
        num_nodes,
        embs,
        all_labels,
        train_losses,
        valid_losses,
    )


# Updated Metrics Calculation Functions
def calculate_accuracy(labels, preds):
    """
    Calculate the accuracy score.

    Args:
        labels (array-like): True labels.
        preds (array-like): Predicted labels.

    Returns:
        float: Accuracy score.
    """
    return accuracy_score(labels, preds)


def calculate_precision(labels, preds):
    """
    Calculate the precision score.

    Args:
        labels (array-like): True labels.
        preds (array-like): Predicted labels.

    Returns:
        float: Precision score.
    """
    return precision_score(labels, preds)


def calculate_recall(labels, preds):
    """
    Calculate the recall score.

    Args:
        labels (array-like): True labels.
        preds (array-like): Predicted labels.

    Returns:
        float: Recall score.
    """
    return recall_score(labels, preds)


def calculate_f1_score(labels, preds):
    """
    Calculate the F1 score.

    Args:
        labels (array-like): True labels.
        preds (array-like): Predicted labels.

    Returns:
        float: F1 score.
    """
    return f1_score(labels, preds)


#     return accuracy, precision, recall, f1
@torch.no_grad()
def compute_metrics(model, loader):
    """
    Compute the evaluation metrics for the model on the given loader.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        loader (torch.utils.data.DataLoader): DataLoader for the dataset to evaluate on.

    Returns:
        dict: Dictionary containing:
            - 'probs': Predicted probabilities (numpy array)
            - 'labels': True labels (numpy array)
    """
    model.eval()
    all_probs, all_labels = [], []
    for data in loader:
        data = data.to(device)
        out, _ = model(data.x, data.edge_index)
        all_probs.append(out.cpu())
        all_labels.append(data.y.cpu())
    probs = torch.cat(all_probs, dim=0).sigmoid().numpy()
    labels = torch.cat(all_labels, dim=0).numpy()

    return {
        "probs": probs,
        "labels": labels,
    }


def experiment_pipeline(
    model_name,
    dataset,
    train_loader,
    valid_loader,
    test_loader,
    num_epochs=100,
    features=2,
):
    """
    Run the full experiment pipeline for a given model and dataset.

    Args:
        model_name (str): Name of the model to use ('GAT', 'GCN', 'GraphSAGE').
        dataset (str): Name of the dataset ('COSS', 'DDINA', 'DDM', etc.).
        train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
        valid_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
        num_epochs (int, optional): Number of training epochs. Defaults to 100.
        features (int, optional): Number of input features. Defaults to 2.

    Returns:
        dict: Dictionary containing experiment results, including model, weights, metrics, ROC, timings, embeddings, labels, and losses.
    """
    num_classes = 1  # Set this according to your dataset
    hidden_channels = 64
    if dataset == "COSS":
        num_features = 14
    else:
        num_features = features

    if model_name == "GAT":
        model = Net(num_features, num_classes)
    elif model_name == "GCN":
        model = GCNNet(num_features, hidden_channels, num_classes)
    elif model_name == "GraphSAGE":
        model = GraphSAGENet(num_features, hidden_channels, num_classes)

    print(f"Running {model_name} Experiment on {dataset}")
    (
        m,
        model_weights,
        fpr,
        tpr,
        _,
        train_times,
        batch_times,
        num_nodes,
        embs,
        labels,
        train_losses,
        valid_losses,
    ) = run_experiment(
        model, train_loader, valid_loader, test_loader, num_epochs=num_epochs
    )
    metrics = compute_metrics(model, test_loader)
    return {
        "model": m,
        "model_weights": model_weights,  # Include the weights in the results
        "metrics": metrics,
        "fpr": fpr,
        "tpr": tpr,
        "train_times": train_times,
        "batch_times": batch_times,
        "num_nodes": num_nodes,
        "embs": embs,
        "labels": labels,
        "train_losses": train_losses,
        "valid_losses": valid_losses,
    }


if __name__ == "__main__":

    # Main script: runs experiments for all datasets and models, saves results to pickle file.
    # label = 1  # Adjust this based on the label you're interested in
    datasets = ["DDINA", "COSS", "DDM"]
    models = ["GAT", "GCN", "GraphSAGE"]
    # num_features = 2  # Set this according to your dataset

    dataset_colors = {"DDINA": "blue", "COSS": "green", "DDM": "red"}

    # Non time dimension related experiments
    results_l = {"DDINA": {}, "COSS": {}, "DDM": {}}
    results_l_wc = {"DDINA": {}, "COSS": {}, "DDM": {}}

    with ProcessPoolExecutor(max_workers=12) as executor:
        future_to_model = {}
        for dataset in datasets:
            train_loader, valid_loader, test_loader = get_split_data_pickle_btc(dataset)
            for model_name in models:
                future = executor.submit(
                    experiment_pipeline,
                    model_name,
                    dataset,
                    train_loader,
                    valid_loader,
                    test_loader,
                    features=14,
                )
                future_to_model[future] = (dataset, model_name)

        for future in as_completed(future_to_model):
            dataset, model_name = future_to_model[future]
            try:
                result = future.result()
                results_l[dataset][model_name] = result
                print(f"Completed {model_name} Experiment on {dataset}")
            except Exception as exc:
                print(
                    f"{model_name} experiment on {dataset} generated an exception: {exc}"
                )

    with open(path.join(PROJECT_ROOT, "data", "results_comparison.pkl"), "wb") as file:
        pickle.dump(results_l, file)
