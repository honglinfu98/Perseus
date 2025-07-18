"""
This script is used to plot the graph with communities. It is used to visualize the communities detected in the graph.
"""

from os import path
import pickle
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.patches as mpatches

# from networkx.algorithms.community import louvain_communities
import matplotlib.pyplot as plt
from perseus.dataset.preprocess.groudtruth_labeling import (
    read_labeling_csv_back_to_dict,
)
from perseus.dataset.preprocess.process import (
    aggregate_data,
    assign_event_ids,
    get_graphs,
    process_dataframe,
    features_engineer,
    compute_weighted_graph_features,
)
from perseus.dataset.preprocess.train_test_validate import (
    # get_train_scored_signals,
    get_btc_test_scored_signals,
    # get_valid_scored_signals,
)
from perseus.dataset.dataset_preparation import combine_features, graph_features
from perseus.settings import PROJECT_ROOT

import re
from datetime import datetime
from matplotlib.legend_handler import HandlerPatch


def extract_elements(data):
    """
    Extracts and processes elements from a nested list structure, cleaning text and formatting timestamps.

    Args:
        data (list): A nested list where each sublist contains tuples with message data. Each tuple is expected to have at least 11 elements, with text at index 10 and timestamp at index 7.

    Returns:
        list: A nested list of tuples, each containing:
            - label (str): Event label (e.g., 'd1', 'd2', ...)
            - channel_name (str): Channel name from item[0]
            - timestamp (str): Formatted timestamp (YYYY-MM-DD HH:MM:SS)
            - text (str): Cleaned message text
            - return_value: Value from item[2]
    """
    results = []
    for sub_index, sub_list in enumerate(data):
        sub_list_results = []  # Create a new list for each sublist's results
        label = (
            f"d{sub_index + 1}"  # Assign a label depending on the index of the sublist
        )
        for item in sub_list:
            # Processing the text to remove emojis, #, and replace \n with space
            text = item[11]
            text = re.sub(
                r"[^\w\s,.]", "", text
            )  # Remove emojis and other non-alphanumeric symbols except commas and periods
            text = text.replace("#", "")  # Remove '#'
            text = text.replace("\n", " ")  # Replace newline characters with spaces
            text = re.sub(
                r"\s+", " ", text
            ).strip()  # Replace multiple spaces with a single space and trim leading/trailing spaces

            # Format timestamp
            timestamp = (
                item[8].strftime("%Y-%m-%d %H:%M:%S")
                if isinstance(item[8], datetime)
                else item[8]
            )

            # Extracting specific indices: label, 0 (channel name), formatted timestamp, processed text, and 2
            extracted_data = (label, item[0], timestamp, text)
            sub_list_results.append(extracted_data)
        results.append(sub_list_results)  # Append the sublist of results
    return results


def generate_latex_table(results):
    """
    Generates a LaTeX table from the processed results for crowd-pump messages and returns.

    Args:
        results (list): Nested list of tuples as returned by extract_elements.

    Returns:
        str: A string containing the LaTeX table code.
    """
    table = r"\begin{table}[!h]" + "\n"
    table += r"\centering" + "\n"
    table += r"\footnotesize" + "\n"
    table += r"\caption{Crowd-pump Messages and Returns}" + "\n"
    table += r"\label{tab: case_study_wrong}" + "\n"
    table += r"\begin{tabularx}{\textwidth}{|c|l|c|X|}" + "\n"
    table += r"\hline" + "\n"
    table += (
        r"\textbf{Events} & \textbf{Telegram Channels} & \textbf{Timestamps} & \textbf{Messages}  \\"
        + "\n"
    )
    table += r"\hline" + "\n"

    last_event = None
    for sub_list in results:
        for index, item in enumerate(sub_list):
            color = (
                "red" if index == 0 else "teal"
            )  # Color for the first item red, others teal
            event_label, channel_name, timestamp, message = item

            # Replace '&' with 'and', and escape underscores
            message = message.replace("&", "and").replace("_", r"\_")
            channel_name = channel_name.replace("_", r"\_")

            # Determine if the event label should be printed
            event_label = event_label if event_label != last_event else ""
            last_event = item[0]  # Update last_event to the current event label

            table += (
                f"\\textcolor{{{color}}}{{{event_label}}} & \\textcolor{{{color}}}{{{channel_name}}} & \\textcolor{{{color}}}{{{timestamp}}} & \\textcolor{{{color}}}{{{message}}} \\\\"
                + "\n"
            )
        table += r"\hline" + "\n"

    table += r"\end{tabularx}" + "\n"
    table += r"\end{table}"

    return table


class HandlerSquare(HandlerPatch):
    def __init__(self, scale=1.0, **kwargs):
        self.scale = scale
        super().__init__(**kwargs)

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        # Compute the center of the legend box.
        center = (xdescent + width / 2, ydescent + height / 2)
        # Compute side length from the available space.
        side = self.scale * min(width, height)
        # Determine the lower-left coordinate so that the square is centered.
        lower_left = (center[0] - side / 2, center[1] - side / 2)
        square = mpatches.Rectangle(
            lower_left,
            side,
            side,
            facecolor=orig_handle.get_facecolor(),
            edgecolor=orig_handle.get_edgecolor(),
            hatch=orig_handle.get_hatch(),
            lw=orig_handle.get_linewidth(),
        )
        self.update_prop(square, orig_handle, legend)
        square.set_transform(trans)
        return [square]


class HandlerCircle(HandlerPatch):
    def __init__(self, scale=1.0, **kwargs):
        """
        Parameters:
            scale (float): A scaling factor to adjust the size of the circle in the legend.
        """
        self.scale = scale
        super().__init__(**kwargs)

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        # Compute the center of the legend box.
        center = (xdescent + width / 2, ydescent + height / 2)
        # Use a radius that fits in the available box, scaled by self.scale.
        r = self.scale * (min(width, height) / 2.0)
        # Create a circle using the full RGBA tuples for colors.
        circle = mpatches.Circle(
            xy=center,
            radius=r,
            facecolor=orig_handle.get_facecolor(),
            edgecolor=orig_handle.get_edgecolor(),
            hatch=orig_handle.get_hatch(),
            lw=orig_handle.get_linewidth(),
        )
        self.update_prop(circle, orig_handle, legend)
        circle.set_transform(trans)
        return [circle]


def community_detection_weighted(P_dict: dict):
    """
    Detects communities in weighted graphs using the Label Propagation algorithm.

    Args:
        P_dict (dict): Dictionary where each key maps to a dict of edge tuples (source, target) and their weights.

    Returns:
        dict: Dictionary mapping each key to a list of sets, each set representing a community of nodes.
    """

    communities_dict = {}

    for key in P_dict.keys():
        G = nx.Graph()
        for (source_node, target_node), weight in P_dict[key].items():
            if weight > 0:
                G.add_edge(source_node, target_node, weight=weight)
        # Label Propagation (returns a generator of sets)
        communities = list(
            nx.algorithms.community.asyn_lpa_communities(G, weight="weight")
        )
        communities_dict[key] = communities

    return communities_dict


def draw_graph_with_communities(
    gs: dict,
    key: str,
    communities_dict: dict,
    id_to_username: dict,
    labels: dict,
    base_node_size=1500,  # Controls the size of the nodes.
    font_size=45,
    arrow_size=40,
    label_distance=0.2,
    community_number_offset=0,  # <--- new parameter
):
    """
    Draws a graph with detected communities, coloring nodes by label and annotating with community numbers.

    Args:
        gs (dict): Dictionary of graphs keyed by coin/event name.
        key (str): The key to select the graph from gs and communities_dict.
        communities_dict (dict): Dictionary mapping keys to lists of communities (sets of nodes).
        id_to_username (dict): Mapping from node IDs to usernames.
        labels (dict): Mapping from node IDs to label values (0 or 1).
        base_node_size (int, optional): Base size for nodes. Default is 1500.
        font_size (int, optional): Font size for labels. Default is 45.
        arrow_size (int, optional): Arrow size for edges. Default is 40.
        label_distance (float, optional): Distance for label offset. Default is 0.2.

    Returns:
        None. Saves the plot as a PDF and displays it.
    """
    # Retrieve graph and communities.
    G = gs[key]
    communities = communities_dict[key]

    # Map nodes to their community index.
    node_to_community = {}
    for idx, community in enumerate(communities):
        for node in community:
            node_to_community[node] = idx

    # Set up figure and axis.
    fig, ax = plt.subplots(figsize=(20, 20))
    pos = nx.circular_layout(G)

    # Draw edges behind nodes.
    nx.draw_networkx_edges(G, pos, arrowsize=arrow_size, width=3, ax=ax)

    # Determine node "radius" based on the layout's x-range.
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    x_range = max(xs) - min(xs)
    default_node_size = 1500
    scale_constant = x_range / (50 * np.sqrt(default_node_size))
    radius = np.sqrt(base_node_size) * scale_constant

    # Draw each node as a circle with a fill color based on its label,
    # and annotate the node with its community number.
    for node in G.nodes():
        x, y = pos[node]
        label_value = labels.get(node, 0)
        # Choose fill color based on the label value.
        if label_value == 0:
            fill_color = "lightblue"
        elif label_value == 1:
            fill_color = "lightcoral"
        else:
            fill_color = "white"  # default color if label is not 0 or 1

        circle = mpatches.Circle(
            (x, y), radius=radius, facecolor=fill_color, edgecolor="black", lw=2
        )
        ax.add_patch(circle)
        # Annotate the node with its community number (starting at 1).
        comm_id = (
            node_to_community[node] + 1 + community_number_offset
        )  # <--- use offset
        ax.text(
            x,
            y,
            str(comm_id),
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=font_size * 1.2,
            color="black",
        )

    # (Optional) Draw additional labels (e.g., usernames) offset from the node.
    label_pos = {
        node: (
            pos[node][0]
            + 6 * label_distance * np.cos(np.arctan2(pos[node][1], pos[node][0])),
            pos[node][1]
            + 1.5 * label_distance * np.sin(np.arctan2(pos[node][1], pos[node][0])),
        )
        for node in [316, 320, 1701, 1738]
    }
    for k in [1505, 309]:
        label_pos[k] = (
            pos[k][0] + 1.5 * label_distance * np.cos(np.arctan2(pos[k][1], pos[k][0])),
            pos[k][1] - 0.5,
        )

    adjusted_labels = {node: id_to_username.get(node, str(node)) for node in G.nodes()}
    nx.draw_networkx_labels(
        G, label_pos, labels=adjusted_labels, font_size=font_size, font_color="black"
    )

    # Build legend for node labels.
    label0_circle = mpatches.Circle(
        (0, 0),
        radius=radius,
        facecolor="lightcoral",
        edgecolor="black",
        lw=2,
        label="True Positive",
    )
    label1_circle = mpatches.Circle(
        (0, 0),
        radius=radius,
        facecolor="lightblue",
        edgecolor="black",
        lw=2,
        label="True Negative",
    )
    leg = ax.legend(
        handles=[label0_circle, label1_circle],
        # title="Node Labels",
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        ncol=2,
        frameon=False,
        fontsize=font_size,
        title_fontsize=font_size,
        handler_map={mpatches.Circle: HandlerCircle(scale=2)},
    )
    ax.add_artist(leg)

    ax.axis("off")
    ax.set_xlim(min(xs) - 2, max(xs) + 2)
    ax.set_ylim(min(ys) - 2, max(ys) + 2)
    fig.tight_layout()
    fig.savefig(path.join(PROJECT_ROOT, "data", "buffer", f"june_{key}_case.pdf"))
    plt.show()


def draw_graph_with_communities_correct_case(
    gs: dict,
    key: str,
    communities_dict: dict,
    id_to_username: dict,
    labels: dict,
    base_node_size=1500,  # Controls the size of the nodes.
    font_size=45,
    arrow_size=40,
    label_distance=0.2,
    community_number_offset=0,  # <--- new parameter
):
    """
    Draws a graph for the 'correct case', coloring nodes by label and annotating with community numbers.

    Args:
        gs (dict): Dictionary of graphs keyed by coin/event name.
        key (str): The key to select the graph from gs and communities_dict.
        communities_dict (dict): Dictionary mapping keys to lists of communities (sets of nodes).
        id_to_username (dict): Mapping from node IDs to usernames.
        labels (dict): Mapping from node IDs to label values (0 or 1).
        base_node_size (int, optional): Base size for nodes. Default is 1500.
        font_size (int, optional): Font size for labels. Default is 45.
        arrow_size (int, optional): Arrow size for edges. Default is 40.
        label_distance (float, optional): Distance for label offset. Default is 0.2.

    Returns:
        None. Saves the plot as a PDF and displays it.
    """
    # Retrieve graph and communities.
    G = gs[key]
    communities = communities_dict[key]

    # Map nodes to their community index.
    node_to_community = {}
    for idx, community in enumerate(communities):
        for node in community:
            node_to_community[node] = idx

    # Set up figure and axis.
    fig, ax = plt.subplots(figsize=(20, 20))
    pos = nx.circular_layout(
        G,
    )

    # Draw edges behind nodes.
    nx.draw_networkx_edges(G, pos, arrowsize=arrow_size, width=3, ax=ax)

    # Determine node "radius" based on the layout's x-range.
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    x_range = max(xs) - min(xs)
    default_node_size = 1500
    scale_constant = x_range / (50 * np.sqrt(default_node_size))
    radius = np.sqrt(base_node_size) * scale_constant

    # Draw each node as a circle with a fill color based on its label,
    # and annotate the node with its community number.
    for node in G.nodes():
        x, y = pos[node]
        label_value = labels.get(node, 0)
        # Choose fill color based on the label value.
        if label_value == 0:
            fill_color = "lightblue"
        elif label_value == 1:
            fill_color = "lightcoral"
        else:
            fill_color = "white"  # default color if label is not 0 or 1

        circle = mpatches.Circle(
            (x, y), radius=radius, facecolor=fill_color, edgecolor="black", lw=2
        )
        ax.add_patch(circle)
        # Annotate the node with its community number (starting at 1).
        comm_id = (
            node_to_community[node] + 1 + community_number_offset
        )  # <--- use offset
        ax.text(
            x,
            y,
            str(comm_id),
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=font_size * 1.2,
            color="black",
        )

    # Draw node labels (e.g., usernames) offset from the node center.
    label_pos = {
        node: (
            pos[node][0]
            + 2.7 * label_distance * np.cos(np.arctan2(pos[node][1], pos[node][0])),
            pos[node][1]
            + 3.7 * label_distance * np.sin(np.arctan2(pos[node][1], pos[node][0])),
        )
        for node in G.nodes()
    }
    # Move 'binance_360' label further away from the center
    for node_id in id_to_username:
        if id_to_username[node_id] == "binance_360" and node_id in label_pos:
            # Increase the offset multiplier for this node
            x, y = pos[node_id]
            angle = np.arctan2(y, x)
            label_pos[node_id] = (
                x + 4 * label_distance * np.cos(angle),  # increase multiplier as needed
                y + 7 * label_distance * np.sin(angle),
            )
    adjusted_labels = {node: id_to_username.get(node, str(node)) for node in G.nodes()}
    nx.draw_networkx_labels(
        G, label_pos, labels=adjusted_labels, font_size=font_size, ax=ax
    )

    # Build legend for node labels.
    label0_circle = mpatches.Circle(
        (0, 0),
        radius=radius,
        facecolor="lightcoral",
        edgecolor="black",
        lw=2,
        label="True Positive",
    )
    label1_circle = mpatches.Circle(
        (0, 0),
        radius=radius,
        facecolor="lightblue",
        edgecolor="black",
        lw=2,
        label="True Negative",
    )
    leg = ax.legend(
        handles=[label0_circle, label1_circle],
        # title="Node Labels",
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        ncol=2,
        frameon=False,
        fontsize=font_size,
        title_fontsize=font_size,
        handler_map={mpatches.Circle: HandlerCircle(scale=2)},
    )
    ax.add_artist(leg)

    ax.axis("off")
    ax.set_xlim(min(xs) - 2, max(xs) + 2)
    ax.set_ylim(min(ys) - 2, max(ys) + 2)
    fig.tight_layout()
    fig.savefig(path.join(PROJECT_ROOT, "data", f"june_{key}_case.pdf"))
    plt.show()


def draw_graph_with_communities_wrong_case(
    gs: dict,
    key: str,
    communities_dict: dict,
    id_to_username: dict,
    predictions: dict,
    base_node_size=1500,
    font_size=45,
    arrow_size=40,
    label_distance=0.4,  # Distance of the label from the node center
    community_number_offset=0,  # <--- new parameter
):
    """
    Draws a graph for the 'wrong case', using rectangles for false predictions and circles for correct ones.

    Args:
        gs (dict): Dictionary of graphs keyed by coin/event name.
        key (str): The key to select the graph from gs and communities_dict.
        communities_dict (dict): Dictionary mapping keys to lists of communities (sets of nodes).
        id_to_username (dict): Mapping from node IDs to usernames.
        predictions (dict): Mapping from node IDs to prediction tuples (true label, predicted label).
        base_node_size (int, optional): Base size for nodes. Default is 1500.
        font_size (int, optional): Font size for labels. Default is 45.
        arrow_size (int, optional): Arrow size for edges. Default is 40.
        label_distance (float, optional): Distance for label offset. Default is 0.4.

    Returns:
        None. Saves the plot as a PDF and displays it.
    """
    G = gs[key]
    communities = communities_dict[key]

    # Map each node to its community.
    node_to_community = {}
    for idx, community in enumerate(communities):
        for node in community:
            node_to_community[node] = idx

    fig, ax = plt.subplots(figsize=(20, 20))
    pos = nx.circular_layout(G)

    # Draw edges behind nodes.
    nx.draw_networkx_edges(G, pos, arrowsize=arrow_size, width=3, ax=ax)

    # Compute node radius.
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    x_range = max(xs) - min(xs)
    radius = (np.sqrt(base_node_size) / np.sqrt(1500)) * (x_range / 50.0)

    # Draw nodes with shapes based on predictions, and annotate each node with its community number.
    for node in G.nodes():
        x, y = pos[node]
        community_index = node_to_community[node]
        prediction = predictions.get(node, (0, 0))  # Default: correct (0,0)

        if prediction in {(1, 0), (0, 1)}:
            # False case: use a rectangle shape
            shape = "rectangle"
            fill_color = "lightcoral" if prediction == (0, 1) else "lightblue"
        else:
            # Correct prediction: use a circle shape
            shape = "circle"
            fill_color = "lightblue"

        if shape == "circle":
            patch = mpatches.Circle(
                (x, y),
                radius=radius,
                facecolor=fill_color,
                edgecolor="black",
                lw=2,
                zorder=3,
            )
        else:
            # Draw a rectangle (square) centered at (x,y) with side length 2*radius.
            patch = mpatches.Rectangle(
                (x - radius, y - radius),
                2 * radius,
                2 * radius,
                facecolor=fill_color,
                edgecolor="black",
                lw=2,
                zorder=3,
            )
        ax.add_patch(patch)
        # Add community number text inside the node.
        ax.text(
            x,
            y,
            str(community_index + 3 + community_number_offset),  # <--- use offset
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=font_size * 1.2,
            color="black",
        )

    # Draw node labels (e.g., usernames) offset from the node center.
    label_pos = {
        node: (
            pos[node][0]
            + 1.6 * label_distance * np.cos(np.arctan2(pos[node][1], pos[node][0])),
            pos[node][1]
            + 2 * label_distance * np.sin(np.arctan2(pos[node][1], pos[node][0])),
        )
        for node in G.nodes()
    }
    adjusted_labels = {node: id_to_username.get(node, str(node)) for node in G.nodes()}
    nx.draw_networkx_labels(
        G, label_pos, labels=adjusted_labels, font_size=font_size, ax=ax
    )

    # Build legend for predictions using rectangle markers.
    false_negative_handle = mpatches.Rectangle(
        (0, 0),
        1,
        1,
        facecolor="lightcoral",
        edgecolor="black",
        lw=2,
        label="False Negative",
    )
    false_positive_handle = mpatches.Rectangle(
        (0, 0),
        1,
        1,
        facecolor="lightblue",
        edgecolor="black",
        lw=2,
        label="False Positive",
    )
    leg2 = ax.legend(
        handles=[false_negative_handle, false_positive_handle],
        title="",
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        ncol=2,
        frameon=False,
        fontsize=font_size,
        title_fontsize=font_size,
        handler_map={mpatches.Rectangle: HandlerSquare(scale=2)},
    )
    leg2.get_frame().set_edgecolor("black")
    ax.add_artist(leg2)

    ax.axis("off")
    ax.set_xlim(min(xs) - 2, max(xs) + 2)
    ax.set_ylim(min(ys) - 2, max(ys) + 2)
    fig.tight_layout()
    fig.savefig(path.join(PROJECT_ROOT, "data", f"june_{key}_case.pdf"))
    plt.show()


# Update the function call in the main block to include id_to_username:
if __name__ == "__main__":
    signals = get_btc_test_scored_signals()
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

    communities_dict = community_detection_weighted(P_theta)
    # Assume id_to_username is available here

    telegram_id = pd.read_csv(path.join(PROJECT_ROOT, "data", "telegram_id.csv"))
    id_to_username = dict(zip(telegram_id.telegram_chat_id, telegram_id.username))

    labels = read_labeling_csv_back_to_dict("test")

    for coin in labels.keys():
        try:
            # Replacing the first element of each tuple in each sublist with the corresponding username
            cascade_labeling[coin] = [
                [
                    tuple(
                        [id_to_username[t[0]] if t[0] in id_to_username else t[0]]
                        + list(t[1:])
                    )
                    for t in sublist
                ]
                for sublist in cascade_labeling[coin]
            ]
        except KeyError:
            # Skip if the coin is not in the cascade_labeling dictionary
            pass

    # load the prediction labels pickle file from the data folder
    with open(path.join(PROJECT_ROOT, "data", "test_predictions.pkl"), "rb") as file:
        predictions = pickle.load(file)

    true_false_labels = {}

    for coin, _ in labels.items():
        true_false_labels[coin] = {}  # Initialize dictionary for each coin
        for i in labels[coin].keys():
            try:
                true_false_labels[coin][i] = (labels[coin][i], predictions[coin][i][0])
            except KeyError:
                # Skip if prediction for this label doesn't exist
                pass

    potential_coins_true = []
    # go over each coin and node in each coin and output the key only wit (0,0) and (1,1), not include (1,0) and (0,1)
    for coin, _ in true_false_labels.items():
        # for node in true_false_labels[coin].keys():
        # Convert true_false_labels[coin][node] to a set for easy checking
        label_set = set(v for i, v in true_false_labels[coin].items())

        # Check if all required tuples are in the set
        if {(1, 1), (0, 0)}.issubset(label_set) and not {(1, 0), (0, 1)}.issubset(
            label_set
        ):
            potential_coins_true.append(coin)

    potential_coins = []
    # go over each coin and node in each coin and output the key with at least (1,0) and (0,1) and (1,1)
    for coin, _ in true_false_labels.items():
        # for node in true_false_labels[coin].keys():
        # Convert true_false_labels[coin][node] to a set for easy checking
        label_set = set(v for i, v in true_false_labels[coin].items())

        # Check if all required tuples are in the set
        if {(1, 0), (0, 1)}.issubset(label_set):
            potential_coins.append(coin)

    # # Plot all true cases
    # for coin in potential_coins_true:
    #     print(f"Plotting true case for coin: {coin}")
    #     draw_graph_with_communities_correct_case(
    #         gs,
    #         coin,
    #         communities_dict,
    #         id_to_username,
    #         labels[coin],
    #         base_node_size=20000,
    #         font_size=55,
    #         arrow_size=120,
    #         label_distance=0.25,
    #     )

    draw_graph_with_communities_correct_case(
        gs,
        "ETC",
        communities_dict,
        id_to_username,
        labels["ETC"],
        base_node_size=20000,
        font_size=55,
        arrow_size=120,
        label_distance=0.25,
        community_number_offset=2,  # ETC: 3,4,...
    )
    # result = [(k[0], k[7], k[-1], k[2]) for i in cascade_labeling["AVAX"] for j in i for k in j]

    draw_graph_with_communities_wrong_case(
        gs,
        "STORJ",
        communities_dict,
        id_to_username,
        predictions["STORJ"],
        base_node_size=20000,
        font_size=55,
        arrow_size=120,
        label_distance=0.25,
        community_number_offset=4,  # STORJ: 5,...
    )

    draw_graph_with_communities(
        gs,
        "SUI",
        communities_dict,
        id_to_username,
        labels["SUI"],
        base_node_size=20000,
        font_size=55,
        arrow_size=120,
        label_distance=0.25,
        community_number_offset=0,  # SUI: 1,2,...
    )

# Assuming 'data' is your complex nested list variable, you would call the function like this:
results = extract_elements(cascade_labeling["ETC"])
latex_table = generate_latex_table(results)
# print(latex_table)


# results = extract_elements(cascade_labeling["SUI"])
# latex_table = generate_latex_table(results)
# print(latex_table)
