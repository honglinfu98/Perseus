import pickle
import numpy as np
import pandas as pd
import torch
from torch.nn import GELU
from torch_geometric.data import Data
from os import path
from perseus.settings import PROJECT_ROOT
from perseus.dataset.preprocess.train_test_validate import (
    get_new_detection,
    get_btc_test_scored_signals,
)
from perseus.dataset.preprocess.process import (
    features_engineer,
    get_graphs,
    process_dataframe,
    aggregate_data,
    assign_event_ids,
    combine_features,
    graph_features,
    out_ego_graph,
    in_ego_graph,
    calculate_effsize_efficiency,
    compute_weighted_graph_features,
)
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy import stats
from sklearn.metrics import f1_score, precision_recall_curve
from perseus.dataset.preprocess.groudtruth_labeling import (
    read_labeling_csv_back_to_dict,
)
from perseus.model.gnn_models import MultiGAT, MultiGraphSAGE
import os
from perseus.dataset.columns import DDM_FEATURE_COLUMNS, DDINA_FEATURE_COLUMNS


def get_detection(
    mode: str,
    data_name: str = "DDM",
    model_name: str = "MultiGAT",
    hidden_channels: int = 128,
    # lr: float = 0.001,
    batch_size: int = 8,
    # results_path: str = "",
) -> tuple[dict, dict]:
    """
    Run detection using a trained GNN model on either test or new data, returning graphs and prediction results.

    Args:
        mode (str): Mode of operation. Must be either 'test' (for test set) or 'new' (for new detection).
        data_name (str, optional): Dataset name. Either 'DDM' or 'DDINA'. Defaults to 'DDM'.
        model_name (str, optional): Model architecture to use. Either 'MultiGAT' or 'MultiGraphSAGE'. Defaults to 'MultiGAT'.
        hidden_channels (int, optional): Number of hidden channels in the GNN model. Defaults to 8.
        lr (float, optional): Learning rate for model training (not used in this function, but kept for compatibility). Defaults to 0.0001.
        batch_size (int, optional): Batch size for model inference. Defaults to 2.
        results_path (str, optional): Path to results file (not used in this function, but kept for compatibility). Defaults to ''.

    Returns:
        tuple[dict, dict]:
            - gs (dict): Dictionary of asset keys to their corresponding NetworkX graphs.
            - output_dict (dict): Dictionary of asset keys to node prediction results (0 or 1) for each node.
    """
    # Load results (unflattened for batch plot)
    with open(
        path.join(PROJECT_ROOT, "data", "new_results_btc_batched.pkl"),
        "rb",
    ) as file:
        batch_results = pickle.load(file)

    # Select the correct experiment
    # Example: DDM, MultiGAT, batch_size=2
    data_name = "DDM"
    model_name = "MultiGAT"
    batch_size = 8

    # Get the result dict for this experiment
    result = batch_results[data_name][model_name][batch_size]

    # Get the model class
    if model_name == "MultiGAT":
        model_cls = MultiGAT
    elif model_name == "MultiGraphSAGE":
        model_cls = MultiGraphSAGE
    else:
        raise ValueError("Unknown model name")

    # Get input feature dimension from your feature columns
    feature_columns = (
        DDM_FEATURE_COLUMNS if data_name == "DDM" else DDINA_FEATURE_COLUMNS
    )
    in_channels = len(feature_columns)
    hidden_channels = result["hidden_channels"]

    # Re-instantiate the model
    model = model_cls(in_channels=in_channels, hidden_channels=hidden_channels)

    # Load the weights
    model.load_state_dict(result["model_weights"])

    # Now model is ready to use!

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # --- Find the best threshold using saved outputs ---
    all_probs = result["metrics"]["probs"]
    all_labels = result["metrics"]["labels"]

    if all_probs.ndim > 1 and all_probs.shape[1] == 1:
        all_probs = all_probs.ravel()
        all_labels = all_labels.ravel()

    precisions, recalls, thresholds = precision_recall_curve(all_labels, all_probs)

    # Set your desired minimum precision
    min_precision = 0.80  # adjust as needed

    # Find indices where precision is above the minimum
    high_precision_indices = np.where(precisions[:-1] >= min_precision)[0]

    if len(high_precision_indices) > 0:
        # Compute F1 for these thresholds
        f1s = []
        for idx in high_precision_indices:
            t = thresholds[idx]
            preds = (all_probs > t).astype(int)
            f1 = f1_score(all_labels, preds, average="macro", zero_division=0)
            f1s.append(f1)
        f1s = np.array(f1s)
        best_idx = high_precision_indices[f1s.argmax()]
        best_threshold = thresholds[best_idx]
        best_macro_f1 = f1s.max()
        print(
            f"Best threshold with precision >= {min_precision}: {best_threshold:.3f}, Macro F1: {best_macro_f1:.3f}, Precision: {precisions[best_idx]:.3f}, Recall: {recalls[best_idx]:.3f}"
        )
    else:
        print(
            f"No threshold found with precision >= {min_precision}. Consider lowering min_precision."
        )

    # --- Get new detection signals ---
    if mode == "test":
        signals = get_btc_test_scored_signals()
    elif mode == "new":
        signals = get_new_detection()
    else:
        raise ValueError("Invalid string input")

    signals["buffer_price_increase"] = signals["price_increase"].apply(
        lambda x: eval(x) if isinstance(x, str) else x
    )
    signals["btc_base_return"] = signals["buffer_price_increase"].apply(
        lambda x: x["price_increase"] if isinstance(x, dict) else np.nan
    )
    signals.drop(columns=["buffer_price_increase"], inplace=True)
    processed_signals = process_dataframe(signals)
    ided_signals = assign_event_ids(processed_signals)
    cascade_buffer, no_nodes_buffer, id_mapping_buffer, cascade_labeling = (
        aggregate_data(ided_signals)
    )
    gs, P_theta = get_graphs(cascade_buffer, no_nodes_buffer, id_mapping_buffer)
    graph_feature = graph_features(gs)
    market_feature = features_engineer(processed_signals)
    weighted_feature = compute_weighted_graph_features(P_theta)
    combine_feature = combine_features(market_feature, graph_feature, weighted_feature)

    # Select feature columns based on data_name
    if data_name == "DDM":
        feature_columns = DDM_FEATURE_COLUMNS
    else:
        feature_columns = DDINA_FEATURE_COLUMNS

    all_predictions = {}

    # Prepare a list of (asset_key, Data, node_ids) for batching
    batch_data = []
    for k, v in gs.items():
        try:
            features_buffer = combine_feature[k]
            features_buffer = features_buffer[
                features_buffer["telegram_chat_id"].isin(v.nodes)
            ]
            features_to_normalize = features_buffer[feature_columns]
            std = features_to_normalize.std()
            mean = features_to_normalize.mean()
            normalized_features = features_to_normalize.copy()
            non_zero_std = std != 0
            normalized_features.loc[:, non_zero_std] = (
                features_to_normalize.loc[:, non_zero_std] - mean[non_zero_std]
            ) / std[non_zero_std]
            normalized_features.loc[:, ~non_zero_std] = 0
            nan_columns = normalized_features.columns[
                normalized_features.isnull().any()
            ].tolist()
            if nan_columns:
                print(f"NaN values detected in key: {k}, Columns: {nan_columns}")
                continue
            node_attributes = torch.tensor(
                normalized_features.values, dtype=torch.float
            )
            id_to_index = {
                telegram_chat_id: index
                for index, telegram_chat_id in enumerate(
                    features_buffer["telegram_chat_id"]
                )
            }
            edge_index = []
            for source_node, target_node in v.edges():
                source = id_to_index[source_node]
                target = id_to_index[target_node]
                edge_index.append([source, target])
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            data = Data(x=node_attributes, edge_index=edge_index)
            node_ids = features_buffer["telegram_chat_id"].tolist()
            batch_data.append((k, data, node_ids))
        except Exception as e:
            print(e)
            continue

    # Process in batches of 8
    batch_size = 2
    for i in range(0, len(batch_data), batch_size):
        batch = batch_data[i : i + batch_size]
        data_list = [item[1].to(device) for item in batch]
        asset_keys = [item[0] for item in batch]
        node_ids_list = [item[2] for item in batch]
        with torch.no_grad():
            output_list = model(data_list)
            for j, output in enumerate(output_list):
                predictions = (output > best_threshold).int()
                all_predictions[asset_keys[j]] = {}
                for idx, pred in enumerate(predictions.cpu().numpy()):
                    node_id = node_ids_list[j][idx]
                    all_predictions[asset_keys[j]][node_id] = pred

    # --- Continue with mastermind detection as before ---
    transformed_predictions = {}
    for asset, nodes in all_predictions.items():
        transformed_predictions[asset] = {
            node: prediction for node, prediction in nodes.items()
        }

    mastermind_set = {}
    for asset, nodes in transformed_predictions.items():
        mastermind_set[asset] = {
            node for node, prediction in nodes.items() if prediction == 1
        }

    output_dict = {}

    for key, marked_nodes in mastermind_set.items():
        if key in gs:
            nodes = [i for i in gs[key].nodes]
            output_dict[key] = {
                node: (1 if node in marked_nodes else 0) for node in nodes
            }
        # Summing up all the values in the dictionaries nested inside the main dictionary.
        total_sum = sum(
            sum(inner_dict.values()) for inner_dict in transformed_predictions.values()
        )
        print(total_sum)

        # Dictionary to hold the count of tokens with 1, 2, 3, etc., '1's
        ones_count = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}

        # Iterating over the tokens
        for token, inner_dict in transformed_predictions.items():
            count_of_ones = list(inner_dict.values()).count(1)
            if (
                1 <= count_of_ones <= 5
            ):  # We are only interested in those tokens with 1 to 5 ones
                ones_count[count_of_ones] += 1

        print(ones_count)

        # top 10 keys have the highest number of ones in trandformed_predictions with their corresponding number of ones
        top_10 = sorted(
            [
                (key, sum(inner_dict.values()))
                for key, inner_dict in transformed_predictions.items()
            ],
            key=lambda x: x[1],
            reverse=True,
        )[:10]
        print(top_10)

    if mode == "test":
        label_mapping = read_labeling_csv_back_to_dict("test")
        combined_predictions = {}
        for asset, nodes in all_predictions.items():
            combined_predictions[asset] = {}
            for node_id, pred in nodes.items():
                # Get label from label_mapping if available, else 0
                asset_labels = label_mapping.get(asset, {})
                label = asset_labels.get(node_id, 0)
                # Save as tuple (prediction, label)
                combined_predictions[asset][node_id] = (int(pred), int(label))

        with open(
            path.join(PROJECT_ROOT, "data", f"{mode}_predictions.pkl"), "wb"
        ) as file:
            pickle.dump(combined_predictions, file)

    return gs, output_dict


if __name__ == "__main__":
    gs, output_dict = get_detection("test")
    gs, output_dict = get_detection("new")
