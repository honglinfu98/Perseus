"""
Module: t2_summary_statistics.py
------------------------------
This script computes and prints summary statistics for cryptocurrency event datasets, including counts of cryptocurrencies, events, graphs, nodes, edges, masterminds, and accomplices. It processes train, validation, and test splits, and supports flexible input of signals and labeling files.
"""

import numpy as np
from perseus.dataset.preprocess.train_test_validate import (
    get_btc_train_scored_signals,
    get_btc_test_scored_signals,
    get_btc_valid_scored_signals,
)
from perseus.dataset.preprocess.process import (
    compute_weighted_graph_features,
    features_engineer,
    get_graphs,
    process_dataframe,
    aggregate_data,
    assign_event_ids,
    graph_features,
    combine_features,
)
from perseus.dataset.preprocess.groudtruth_labeling import (
    read_labeling_csv_back_to_dict,
)
from perseus.dataset.dataset_preparation import get_split_data_pickle_btc_noloader


def generate_summary_statistics(
    train_signals,
    valid_signals,
    test_signals,
    train_label_file="train",
    valid_label_file="valid",
    test_label_file="test",
    direct_model="DDINA",
    weighted_model="DDM",
):
    """
    Generate summary statistics for cryptocurrency event datasets.

    Parameters:
        train_signals: List or DataFrame of training signals.
        valid_signals: List or DataFrame of validation signals.
        test_signals: List or DataFrame of test signals.
        train_label_file: Filename or identifier for training labels.
        valid_label_file: Filename or identifier for validation labels.
        test_label_file: Filename or identifier for test labels.
        direct_model: Model name for direct graph data.
        weighted_model: Model name for weighted graph data.

    Returns:
        dict: Summary statistics including counts of cryptocurrencies, events, graphs, nodes, edges, masterminds, and accomplices for each split.
    """
    datasets = [train_signals, valid_signals, test_signals]
    gs_ls = []
    features_ls = []
    cascade_ls = []
    P_theta_ls = []

    for dataset in datasets:
        processed_signals = process_dataframe(dataset)
        ided_signals = assign_event_ids(processed_signals)
        cascade, no_nodes, id_mapping, cascade_labeling = aggregate_data(ided_signals)
        gs, P_theta = get_graphs(cascade, no_nodes, id_mapping)
        graph_feature = graph_features(gs)
        market_feature = features_engineer(processed_signals)
        weighted_feature = compute_weighted_graph_features(P_theta)
        combine_feature = combine_features(
            market_feature, graph_feature, weighted_feature
        )
        gs_ls.append(gs)
        features_ls.append(combine_feature)
        cascade_ls.append(cascade)
        P_theta_ls.append(P_theta)

    label_mapping_ls = [
        read_labeling_csv_back_to_dict(train_label_file),
        read_labeling_csv_back_to_dict(valid_label_file),
        read_labeling_csv_back_to_dict(test_label_file),
    ]
    # Replace None with empty dicts for robustness
    label_mapping_ls = [lm if lm is not None else {} for lm in label_mapping_ls]

    for i in range(3):
        common_keys = set(label_mapping_ls[i].keys())
        gs_ls[i] = {key: gs_ls[i][key] for key in common_keys if key in gs_ls[i]}
        features_ls[i] = {
            key: features_ls[i][key] for key in common_keys if key in features_ls[i]
        }
        P_theta_ls[i] = {
            key: P_theta_ls[i][key] for key in common_keys if key in P_theta_ls[i]
        }
        label_mapping_ls[i] = {
            k: v for k, v in label_mapping_ls[i].items() if k in common_keys
        }

    direct = get_split_data_pickle_btc_noloader(direct_model)
    weighted = get_split_data_pickle_btc_noloader(weighted_model)

    stats = {
        "number_of_cryptocurrency": [
            len(cascade_ls[0]),
            len(cascade_ls[1]),
            len(cascade_ls[2]),
        ],
        "crowd_pumps": [len(datasets[0]), len(datasets[1]), len(datasets[2])],
        "crowd_pumps_event": [
            sum([len(v) for v in cascade_ls[0].values()]),
            sum([len(v) for v in cascade_ls[1].values()]),
            sum([len(v) for v in cascade_ls[2].values()]),
        ],
        "mastermind": [
            sum([sum(np.array(i.y[:, 0])) for i in direct[0]]),
            sum([sum(np.array(i.y[:, 0])) for i in direct[1]]),
            sum([sum(np.array(i.y[:, 0])) for i in direct[2]]),
        ],
        "accomplices": [
            sum(
                [len(np.array(i.y[:, 0])) - sum(np.array(i.y[:, 0])) for i in direct[0]]
            ),
            sum(
                [len(np.array(i.y[:, 0])) - sum(np.array(i.y[:, 0])) for i in direct[1]]
            ),
            sum(
                [len(np.array(i.y[:, 0])) - sum(np.array(i.y[:, 0])) for i in direct[2]]
            ),
        ],
        "number_of_graphs": [len(gs_ls[0]), len(gs_ls[1]), len(gs_ls[2])],
        "number_of_nodes": [
            sum([len(v.nodes()) for v in gs_ls[0].values()]),
            sum([len(v.nodes()) for v in gs_ls[1].values()]),
            sum([len(v.nodes()) for v in gs_ls[2].values()]),
        ],
        "number_of_edges_directed": [
            sum([len(np.array(i.edge_index[0, :])) for i in direct[0]]),
            sum([len(np.array(i.edge_index[0, :])) for i in direct[1]]),
            sum([len(np.array(i.edge_index[0, :])) for i in direct[2]]),
        ],
        "number_of_edges_weighted": [
            sum([len(np.array(i.edge_index[0, :])) for i in weighted[0]]),
            sum([len(np.array(i.edge_index[0, :])) for i in weighted[1]]),
            sum([len(np.array(i.edge_index[0, :])) for i in weighted[2]]),
        ],
    }
    return stats


if __name__ == "__main__":
    train_signals = get_btc_train_scored_signals()
    valid_signals = get_btc_valid_scored_signals()
    test_signals = get_btc_test_scored_signals()
    stats = generate_summary_statistics(train_signals, valid_signals, test_signals)
    for k, v in stats.items():
        print(f"{k}: {v}")
