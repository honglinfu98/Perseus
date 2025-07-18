import pickle
import numpy as np
import pandas as pd
import torch
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

from perseus.evaluation.detection import get_detection


# --- Example: Run your analysis/plotting functions as before ---
def aggregate_and_compare_combined(
    gs_ls: dict, label_mapping_ls: dict, font_size: int, tick_size: int, bw: float
):
    """
    Aggregates node-level graph metrics by group (Accomplices vs. Mastermind), performs statistical comparison, and plots distributions.

    Args:
        gs_ls (dict): Dictionary mapping asset keys to their corresponding NetworkX graphs. Each graph represents a network of nodes (e.g., users or entities).
        label_mapping_ls (dict): Dictionary mapping asset keys to dictionaries of node labels. Each inner dictionary maps node IDs to group labels (0 for Accomplices, 1 for Mastermind).
        font_size (int): Font size for plot labels and titles.
        tick_size (int): Font size for axis ticks in plots.
        bw (float): Bandwidth adjustment for kernel density estimation in seaborn plots.

    Returns:
        tuple:
            - ttest_results (dict): Dictionary mapping metric names to t-test results (scipy.stats.stats.Ttest_indResult) comparing the two groups.
            - metrics_by_group (dict): Dictionary mapping metric names to lists of values for each group (index 0: Accomplices, index 1: Mastermind).
    """
    # metrics_by_group: stores metric values for each group (0: Accomplices, 1: Mastermind)
    metrics_by_group = {
        "betweenness_centrality": [[], []],
        "closeness_centrality": [[], []],
        "pagerank": [[], []],
        "in_ratio": [[], []],
        "Out Ratio": [[], []],
        "out_nodes": [[], []],
        "Effective Size": [[], []],
        "Efficiency": [[], []],
        "Density": [[], []],
        "Clustering Coefficient": [[], []],
    }
    for key, graph in gs_ls.items():
        labels = label_mapping_ls.get(key, {})  # node_id -> group label (0 or 1)
        betweenness_centrality = nx.betweenness_centrality(graph)
        closeness_centrality = nx.closeness_centrality(graph)
        pagerank = nx.pagerank(graph)
        clustering_coeff = nx.clustering(graph)
        for node in graph.nodes():
            group = labels.get(node)
            if group is not None:
                # Append metrics to the appropriate group list
                metrics_by_group["betweenness_centrality"][group].append(
                    betweenness_centrality.get(node, 0)
                )
                metrics_by_group["closeness_centrality"][group].append(
                    closeness_centrality.get(node, 0)
                )
                metrics_by_group["pagerank"][group].append(pagerank.get(node, 0))
                metrics_by_group["Clustering Coefficient"][group].append(
                    clustering_coeff.get(node, 0)
                )
                in_ego = in_ego_graph(graph, node)
                out_ego = out_ego_graph(graph, node)
                in_ratio = (len(in_ego.nodes) - 1) / len(graph.nodes)
                out_ratio = (len(out_ego.nodes) - 1) / len(graph.nodes)
                eff_size, efficiency = calculate_effsize_efficiency(graph, node)
                density = nx.density(out_ego)
                metrics_by_group["in_ratio"][group].append(in_ratio)
                metrics_by_group["Out Ratio"][group].append(out_ratio)
                metrics_by_group["out_nodes"][group].append(len(out_ego.nodes))
                metrics_by_group["Effective Size"][group].append(eff_size)
                metrics_by_group["Efficiency"][group].append(efficiency)
                metrics_by_group["Density"][group].append(density)
    ttest_results = {}
    for metric, groups in metrics_by_group.items():
        # Perform Welch's t-test between the two groups for each metric
        ttest_results[metric] = stats.ttest_ind(
            groups[0], groups[1], equal_var=False, nan_policy="omit"
        )

    # --- Plotting logic from mastermind_detection.py ---
    metrics_to_plot = [
        "Efficiency",
        "Out Ratio",
    ]
    plt.rc("font", size=font_size)
    plt.rc("axes", titlesize=font_size, labelsize=font_size)
    plt.rc("xtick", labelsize=tick_size)
    plt.rc("ytick", labelsize=tick_size)
    plt.rc("legend", fontsize=font_size)
    group_colors = {"Accomplices": "lightblue", "Mastermind": "lightcoral"}
    for metric in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(8, 8))
        vertical_lines_info = []
        if metric == "Efficiency":
            groups_info = [
                (metrics_by_group[metric][1], "Mastermind"),
                (metrics_by_group[metric][0], "Accomplices"),
            ]
        else:  # Clustering Coefficient
            groups_info = [
                (metrics_by_group[metric][0], "Accomplices"),
                (metrics_by_group[metric][1], "Mastermind"),
            ]
        for data, group_label in groups_info:
            mean_val = np.mean(data)
            color = group_colors[group_label]
            if metric == "Efficiency":
                legend_label = (
                    group_label if group_label == "Accomplices" else "_nolegend_"
                )
            else:
                legend_label = (
                    group_label if group_label == "Mastermind" else "_nolegend_"
                )
            sns.kdeplot(
                data,
                ax=ax,
                label=legend_label,
                bw_adjust=bw,
                clip=(0, np.inf),
                linewidth=5,
                color=color,
            )
            ax.axvline(mean_val, color=color, linestyle="--", linewidth=2)
            vertical_lines_info.append((mean_val, color, group_label))
        y_max = ax.get_ylim()[1]
        for mean_val, color, group_label in vertical_lines_info:
            offset = (5, 0)
            horizontal_alignment = "left"
            ax.annotate(
                f"{mean_val:.2f}",
                xy=(mean_val, y_max * 0.95),
                xytext=offset,
                textcoords="offset points",
                color=color,
                weight="bold",
                va="center",
                ha=horizontal_alignment,
            )
        ax.set_xlabel(metric, fontsize=font_size)
        ax.set_ylabel("", fontsize=font_size)
        for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
            tick_label.set_fontweight("normal")
        ax.tick_params(axis="both", which="major", labelsize=font_size)
        ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.5)
        ax.legend(
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=1,
            prop={"weight": "normal", "size": font_size},
            title_fontsize=font_size,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.9])
        plt.savefig(path.join(PROJECT_ROOT, "data", f"distribution_{metric}.pdf"))
        plt.show()
        plt.close(fig)
    return ttest_results, metrics_by_group


def assign_significance(p: float):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""


if __name__ == "__main__":

    # --- Run the pooled analysis ---
    gs, output_dict = get_detection("new")
    results, c = aggregate_and_compare_combined(gs, output_dict, 28, 28, 0.9)

    data = {
        "Metric": [],
        "Statistic": [],
        "P-value": [],
        "Degrees of Freedom": [],
        "Significance": [],
        "Group 0 Mean": [],
        "Group 1 Mean": [],
    }
    for metric, result in results.items():
        data["Metric"].append(metric)
        data["Statistic"].append(result.statistic)
        data["P-value"].append(f"{result.pvalue:.3g}")
        data["Degrees of Freedom"].append(getattr(result, "df", None))
        data["Significance"].append(assign_significance(result.pvalue))
        mean_group0 = np.mean(c[metric][0]) if c[metric][0] else np.nan
        mean_group1 = np.mean(c[metric][1]) if c[metric][1] else np.nan
        data["Group 0 Mean"].append(mean_group0)
        data["Group 1 Mean"].append(mean_group1)
    df_results = pd.DataFrame(data)
    df_results.set_index("Metric", inplace=True)
    print(df_results)

    print("Detection complete. Example output_dict:", output_dict)
