"""
This script is used to process the scored signals and create summary for the data
"""

from datetime import timedelta
import json
from collections import defaultdict
import pandas as pd
import networkx as nx
from perseus.dataset.preprocess.train_test_validate import (
    get_btc_train_scored_signals,
    get_btc_test_scored_signals,
    get_btc_valid_scored_signals,
)
from perseus.dataset.preprocess.DANI import DANI


def graph_features(gs: dict):
    """
    This function is used to extract graph features from the graph
    """
    dfs = {}

    # Looping through each key in the scores dictionary
    for key in gs.keys():
        # Creating lists to store node_ids, in-ego ratios, and out-ego ratios
        node_ids = []
        in_ratios = []
        out_ratios = []
        out_nodes = []
        eff_sizes = []
        efficiencies = []
        density = []
        clustering_coeffs = []
        closeness_centralities = []
        betweenness_centrality = []
        pagerank = []
        # degree_centrality = nx.degree_centrality(graph)
        # closeness_centrality = nx.closeness_centrality(graph)
        # pagerank = nx.pagerank(graph)

        for i in gs[key].nodes():
            node_id = i
            node_ids.append(node_id)

            # Calculating the in-ego ratio
            inn = len(in_ego_graph(gs[key], node_id).nodes) / len(gs[key].nodes)
            in_ratios.append(inn)

            # Calculating the out-ego ratio
            outt = len(out_ego_graph(gs[key], node_id).nodes) / len(gs[key].nodes)
            out_ratios.append(outt)

            out_nodes.append(len(out_ego_graph(gs[key], node_id).nodes))

            # Calculate the density of the ego network
            den = nx.density(out_ego_graph(gs[key], node_id))
            density.append(den)

            eff_size, efficiency = calculate_effsize_efficiency(gs[key], node_id)
            eff_sizes.append(eff_size)
            efficiencies.append(efficiency)
            # Calculate clustering coefficient for the node
            clustering_coeff = nx.clustering(gs[key], node_id)
            clustering_coeffs.append(clustering_coeff)
            # calcualte the cloness centrality
            closeness_centrality = nx.closeness_centrality(gs[key], node_id)
            closeness_centralities.append(closeness_centrality)
            # calculate the betweenness centrality
            betweenness_centralit = nx.betweenness_centrality(gs[key])[node_id]
            betweenness_centrality.append(betweenness_centralit)
            # calculate the pagerank
            pr = nx.pagerank(gs[key])[node_id]
            pagerank.append(pr)

        # Creating a DataFrame for each key and storing it in the dfs dictionary
        dfs[key] = pd.DataFrame(
            {
                "telegram_chat_id": node_ids,
                "ego_in_ratio": in_ratios,
                "ego_out_ratio": out_ratios,
                "ego_out_nodes": out_nodes,
                "eff_size": eff_sizes,
                "efficiency": efficiencies,
                "density": density,
                "clustering_coeff": clustering_coeffs,
                "closeness_centrality": closeness_centralities,
                "pagerank": pagerank,
                "betweenness_centrality": betweenness_centrality,
            }
        )

    return dfs


def compute_weighted_graph_features(edge_weights: dict):
    dfs = {}

    for key in edge_weights.keys():
        # Build the weighted directed graph
        G = nx.DiGraph()
        for k, w in edge_weights[key].items():
            if w != 0:
                u = k[0]
                v = k[1]
                G.add_edge(u, v, weight=w)

        # Prepare lists to store results
        node_ids = []

        weighted_in_ratios = []
        weighted_out_ratios = []
        out_weights = []
        closeness_centrality = []
        betweenness_centrality = []
        pagerank = []
        efficiency = []
        eff_size = []
        clustering_coeffs = []
        weighted_density = []

        # Compute total edge weight W
        W = sum(edge_weights[key].values()) or 1  # Avoid division by zero

        # Calculate closeness centrality with distance based on inverse weight
        closeness_centrality_dict = nx.closeness_centrality(G, distance="weight")
        betweenness_centrality_dict = nx.betweenness_centrality(G, weight="weight")
        pagerank_dict = nx.pagerank(G, weight="weight")
        clustering_coeff_dict = nx.clustering(G, weight="weight")

        # Calculate metrics for each node
        for node in G.nodes():
            node_ids.append(node)

            # Append weighted in and out ratios using the in/out ego graph
            weighted_in_ratio = in_ego_graph_weighted(G, node).size(weight="weight") / W
            weighted_out_ratio = (
                out_ego_graph_weighted(G, node).size(weight="weight") / W
            )

            weighted_in_ratios.append(weighted_in_ratio)
            weighted_out_ratios.append(weighted_out_ratio)

            out_weights.append(out_ego_graph_weighted(G, node).size(weight="weight"))

            # Append closeness centrality
            closeness_centrality.append(closeness_centrality_dict[node])
            # Append betweenness centrality
            betweenness_centrality.append(betweenness_centrality_dict[node])
            # Append PageRank values
            pagerank.append(pagerank_dict[node])
            clustering_coeffs.append(clustering_coeff_dict[node])

            # Calculate weighted density using the provided formula

            # Ego networks: For clustering coefficient, effective size, efficiency, and density
            ego_graph = nx.ego_graph(G, node, radius=1, center=True, undirected=False)
            weighted_den = calculate_weighted_density(ego_graph)
            weighted_density.append(weighted_den)

            # Calculate effective size and efficiency
            eff_size_val, efficiency_val = calculate_effsize_efficiency_weighted(
                G, node
            )
            eff_size.append(eff_size_val)
            efficiency.append(efficiency_val)

        # Create DataFrame for the results
        dfs[key] = pd.DataFrame(
            {
                "telegram_chat_id": node_ids,
                "ego_weighted_in_ratio": weighted_in_ratios,
                "ego_weighted_out_ratio": weighted_out_ratios,
                "ego_out_weights": out_weights,
                "weighted_closeness_centrality": closeness_centrality,
                "weighted_betweenness_centrality": betweenness_centrality,
                "weighted_pagerank": pagerank,
                "ego_weighted_eff_size": eff_size,
                "ego_weighted_efficiency": efficiency,
                "weighted_clustering_coefficient": clustering_coeffs,
                "ego_weighted_density": weighted_density,
            }
        )

    return dfs


def calculate_effsize_efficiency(G, ego):
    """
    This function is used to calculate the effective size and efficiency of a node in a graph
    """
    # Get the ego network
    ego_net = nx.ego_graph(G, ego, undirected=False)

    # Get alters in the ego network (excluding ego)
    alters = set(ego_net.nodes()) - {ego}
    num_alters = len(alters)
    avg_degree = 0  # Default to 0
    if num_alters > 0:
        avg_degree = (
            sum(
                [1 for i in alters for j in alters if i != j and ego_net.has_edge(i, j)]
            )
            / num_alters
        )

    # Calculate effective size
    eff_size = num_alters - avg_degree
    # Calculate efficiency
    efficiency = eff_size / num_alters if num_alters > 0 else 0

    return eff_size, efficiency


def calculate_effsize_efficiency_weighted(G, ego):
    """
    This function calculates the effective size and efficiency of a node in a weighted directed graph.
    Effective size measures the number of alters ego has that are not connected to each other, weighted by the strength of the connections.
    Efficiency is the ratio of the effective size to the number of alters, indicating how directly connected the ego is to its alters.
    """
    # Get the ego network, considering it as directed and including edge weights
    ego_net = nx.ego_graph(G, ego, undirected=False)

    # Get alters in the ego network, excluding the ego itself
    alters = set(ego_net.nodes()) - {ego}
    num_alters = len(alters)

    # Initialize average degree, considering weights
    avg_degree = 0
    if num_alters > 0:
        total_weighted_degree = 0
        for n in alters:
            # Sum weights of outgoing edges from each alter, excluding any edges pointing back to ego
            node_weighted_degree = sum(
                weight for _, _, weight in ego_net.edges(n, data="weight") if _ != ego
            )
            # Subtract the weight of any edge from ego to this alter, if it exists
            if ego_net.has_edge(ego, n):
                node_weighted_degree -= ego_net[ego][n].get("weight", 0)
            total_weighted_degree += node_weighted_degree

        avg_degree = total_weighted_degree / num_alters

    # Calculate effective size as the difference between the number of alters and the average weighted degree
    eff_size = num_alters - avg_degree
    # Calculate efficiency as the ratio of the effective size to the number of alters
    efficiency = eff_size / num_alters if num_alters > 0 else 0

    return eff_size, efficiency


# Function to calculate weighted density
def calculate_weighted_density(G):
    """
    Calculate the weighted density of a graph, which is defined as the sum of the weights of all edges
    divided by the number of possible edges in the graph (|V| * (|V| - 1)).
    """
    total_weight = G.size(weight="weight")  # Sum of the weights of all edges
    num_nodes = len(G.nodes())

    # Number of possible directed edges between nodes (no self-loops)
    possible_edges = num_nodes * (num_nodes - 1)

    # Weighted density is the total edge weight divided by the number of possible edges
    return total_weight / possible_edges if possible_edges > 0 else 0


# Function to extract the out-ego network for weighted graphs
def out_ego_graph_weighted(G, node, radius=1):
    """
    Extract the out-ego network of a specified node in a directed weighted graph.
    """
    # Extract the out-ego network, preserving weights
    out_ego = nx.ego_graph(G, node, radius=radius, undirected=False, distance="weight")
    return out_ego


# Function to extract the in-ego network for weighted graphs
def in_ego_graph_weighted(G, node, radius=1):
    """
    Extract the in-ego network of a specified node in a directed weighted graph.
    """
    # Reverse the graph to get the in-ego network, preserving weights
    G_reverse = G.reverse(copy=True)
    in_ego = nx.ego_graph(
        G_reverse, node, radius=radius, undirected=False, distance="weight"
    )
    return in_ego


def out_ego_graph(G, node, radius=1):
    """
    Extract the out-ego network of a specified node in a directed graph.
    """
    # Extract the out-ego network
    out_ego = nx.ego_graph(G, node, radius=radius)
    return out_ego


def in_ego_graph(G, node, radius=1):
    """
    Extract the in-ego network of a specified node in a directed graph.
    """
    # Reverse the graph
    G_reverse = G.reverse(copy=True)
    # Extract the in-ego network
    in_ego = nx.ego_graph(G_reverse, node, radius=radius)
    return in_ego


# merge three dfs market_features: dict, graph_features: dict, weighted_features: dict like combine_features
def combine_features(
    market_features: dict, graph_features: dict, weighted_features: dict
):
    """
    This function is used to combine the market and graph features
    """
    combined_data = {}

    for key in market_features.keys():
        # Check if the key exists in dfs and 'telegram_chat_id' exists in both DataFrames
        if (
            key in graph_features
            and "telegram_chat_id" in market_features[key].columns
            and "telegram_chat_id" in graph_features[key].columns
            and "telegram_chat_id" in weighted_features[key].columns
        ):
            # Perform an inner join on 'telegram_chat_id'
            combined_df = pd.merge(
                market_features[key],
                graph_features[key],
                on="telegram_chat_id",
                how="inner",
            )
            combined_df = pd.merge(
                combined_df,
                weighted_features[key],
                on="telegram_chat_id",
                how="inner",
            )
        else:
            # If key is not in dfs or 'telegram_chat_id' is missing in either DataFrame, use an empty DataFrame
            combined_df = pd.DataFrame()

        combined_data[key] = combined_df

    return combined_data


def relabel_edges(d: dict, mapping: dict):
    """
    Function to relabel edges based on new_to_id mapping
    """
    new_d = defaultdict(float)
    for (src, dst), weight in d.items():
        # Relabel src and dst using the mapping
        new_src = mapping.get(src, src)  # Use original if not in mapping
        new_dst = mapping.get(dst, dst)  # Use original if not in mapping
        new_d[(new_src, new_dst)] = weight
    return new_d


def assign_event_ids(df: pd.DataFrame):
    """
    Assigns event IDs to the pump signals based on the session timeout
    """

    # Initialize the event ID
    event_id = 0

    # Group the data by commodity
    groups = df.groupby(["commodity", "position"])

    # For each group...
    for _, group in groups:
        # Sort the group by source_datetime
        group = group.sort_values("start_date")
        session_timeout = group["time_diff"].quantile(0.95)
        if session_timeout > timedelta(hours=72):
            session_timeout = timedelta(hours=72)

        # Track the previous timestamp for the first row as the first session start time
        prev_time = group.iloc[0]["start_date"]

        # Assign the first event ID to the first row
        df.loc[group.index[0], "event_id"] = event_id

        # For the remaining rows in the group...
        for i, row in group.iloc[1:].iterrows():
            # If the difference between the current row's source_datetime and the previous timestamp exceeds the session_timeout...
            if row["start_date"] - prev_time > session_timeout:
                # Increment the event ID
                event_id += 1

            # Assign the current event ID to the row
            df.loc[i, "event_id"] = event_id

            # Update the previous timestamp
            prev_time = row["start_date"]

        # Increase the event ID for the next commodity (to ensure the next commodity starts with a new event ID)
        event_id += 1

    return df


def process_dataframe(signals_df: pd.DataFrame):
    """
    Process the dataframe to extract and generate the features
    """

    # Parsing 'price_increase' and extracting the values
    df = signals_df[signals_df["price_increase"] != "TRADE DATA NOT AVAILABLE"]
    df["parsed_price_increase"] = df["price_increase"].apply(json.loads)
    df["increase_percentage"] = df["parsed_price_increase"].apply(
        lambda x: x.get("price_increase", None)
    )
    df["start_price"] = df["parsed_price_increase"].apply(lambda x: x.get("from", None))
    df["end_price"] = df["parsed_price_increase"].apply(lambda x: x.get("to", None))
    # create a new column with value "long" if start_price is greater than end_price otherwise "short"
    df["position"] = df.apply(
        lambda x: "long" if x["start_price"] < x["end_price"] else "short", axis=1
    )
    df["parsed_rate"] = df["targets_achieved_rate"].apply(json.loads)
    df["targets_achieved"] = df["parsed_rate"].apply(
        lambda x: x.get("targets_achieved", None)
    )
    df["total_targets"] = df["parsed_rate"].apply(
        lambda x: x.get("total_targets", None)
    )
    df["parsed_duration"] = df["duration"].apply(json.loads)
    df["duration_min"] = df["parsed_duration"].apply(
        lambda x: x.get("duration_min", None)
    )
    df["start_date"] = df["parsed_duration"].apply(lambda x: x.get("start_date", None))
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = df["parsed_duration"].apply(lambda x: x.get("end_date", None))
    df["end_date"] = pd.to_datetime(df["end_date"])
    df["speed"] = df["increase_percentage"] / df["duration_min"]
    df = df.sort_values("start_date", ascending=True)

    # Calculate the time difference between consecutive rows for the same entity_id and signal type
    df["time_diff"] = df.groupby(["position", "commodity"])["start_date"].diff()

    df["btc_base_return"] = df["btc_base_return"].fillna(df["increase_percentage"])

    return df


def aggregate_data(df: pd.DataFrame):
    """
    This function aggregates the data based on the commodity and the event_id
    """
    df_grouped = df.sort_values("start_date").groupby(["commodity", "event_id"])

    intermediate_results = {}
    unique_chat_ids_per_commodity = {}
    mappings = {}

    for (commodity, _), group in df_grouped:
        dates = pd.to_datetime(group["start_date"])
        delta_minutes = (dates - dates.iloc[0]).dt.total_seconds() / 60

        # Use drop_duplicates to keep only the first appearing telegram_chat_id
        group = group.drop_duplicates(subset="telegram_chat_id", keep="first")
        tuples = list(
            zip(
                group["telegram_chat_id"],
                delta_minutes,
                group["increase_percentage"],
                group["btc_base_return"],
                group["position"],
                group["targets_achieved"],
                group["total_targets"],
                group["duration_min"],
                group["start_date"],
                group["end_date"],
                group["speed"],
                group["message_text"],
            )
        )

        if len(tuples) >= 2:
            if commodity not in intermediate_results:
                intermediate_results[commodity] = []

            intermediate_results[commodity].append(tuples)

            # Track unique telegram_chat_id per commodity
            if commodity not in unique_chat_ids_per_commodity:
                unique_chat_ids_per_commodity[commodity] = set()

            unique_chat_ids_per_commodity[commodity].update(
                group["telegram_chat_id"].tolist()
            )

    final_results = {}

    for commodity, lists_of_tuples in intermediate_results.items():
        unique_ids = unique_chat_ids_per_commodity[commodity]
        id_mapping = {
            old_id: new_id for new_id, old_id in enumerate(sorted(unique_ids), start=0)
        }

        # Inverse mapping from new_id to old_id
        new_id_to_old = {v: k for k, v in id_mapping.items()}

        # Save the mappings for future use
        if commodity not in mappings:
            mappings[commodity] = {}
        mappings[commodity]["id_to_new"] = id_mapping
        mappings[commodity]["new_to_id"] = new_id_to_old  # Include the inverse mapping

        mapped_list = []
        for tuple_list in lists_of_tuples:
            data_dict = {
                t[0]: t[1] for t in tuple_list
            }  # Convert tuple_list to a dictionary first
            data_dict["T"] = tuple_list[-1][1]  # Include the 'T' key
            mapped_dict = {id_mapping.get(k, k): v for k, v in data_dict.items()}
            mapped_list.append(mapped_dict)

        final_results[commodity] = mapped_list

    unique_counts = {
        commodity: len(chat_ids)
        for commodity, chat_ids in unique_chat_ids_per_commodity.items()
    }

    return final_results, unique_counts, mappings, intermediate_results


def features_engineer(df: pd.DataFrame):
    """
    This function is used to engineer market features for the signals
    """
    grouped_data = (
        df.groupby(["commodity", "telegram_chat_id"])
        .agg(
            {
                "increase_percentage": "mean",
                "btc_base_return": "mean",
                "speed": "mean",
                "id": "count",
                "chat_crowd_score": "last",
                "targets_achieved": "sum",
                "total_targets": "sum",
            }
        )
        .reset_index()
        .rename(
            columns={
                "increase_percentage": "average_increase_percentage",
                "btc_base_return": "average_btc_base_return",
                "speed": "average_speed",
                "id": "number_of_signals",
                "chat_crowd_score": "latest_chat_crowd_score",
                "targets_achieved": "sum_targets_achieved",
                "total_targets": "sum_total_targets",
            }
        )
    )

    # Organize the results into a dictionary
    result_dict = {}
    for commodity in grouped_data["commodity"].unique():
        result_dict[commodity] = grouped_data[grouped_data["commodity"] == commodity][
            [
                "telegram_chat_id",
                "average_increase_percentage",
                "average_btc_base_return",
                "average_speed",
                "number_of_signals",
                "latest_chat_crowd_score",
                "sum_targets_achieved",
                "sum_total_targets",
            ]
        ]

    for commodity, df in result_dict.items():
        result_dict[commodity]["rating"] = (
            result_dict[commodity]["sum_targets_achieved"]
            / result_dict[commodity]["sum_total_targets"]
        )

    return result_dict


def get_graphs(cascade: dict, no_nodes: dict, id_mapping: dict):
    """
    This function creates the graphs for the cascadeX
    :param cascade: the cascade
    :param no_nodes: the number of nodes for each commodity
    :param id_mapping: the mapping between the commodity and the id
    :return: the graphs, the weight ordered edges, the edge list, and relabeled edge list
    """
    ensure_graph_learned = {}
    # Filter the no_nodes that have more than 3 nodes and have more than no nodes cascade
    for key, value in no_nodes.items():
        if value > 3:  #  and value < len(cascade[key])
            ensure_graph_learned[key] = value

    # Create the graphs for each commodity using moer_than_three
    graphs = {}

    P_dict = {}
    P_theta = {}
    for key, value in ensure_graph_learned.items():
        print(f"Creating graph for {key} with {value} nodes")
        graphs[key], P_dict[key], P_theta[key] = DANI(
            ensure_graph_learned[key], cascade[key]
        )
        graphs[key] = nx.relabel_nodes(graphs[key], id_mapping[key]["new_to_id"])

    # Loop over each key in P_com and apply the mapping
    for key in P_theta:
        if key in id_mapping:
            # Extract new_to_id mapping for the current key
            new_to_id = id_mapping[key]["new_to_id"]
            # Relabel edges in the current dictionary using the new_to_id mapping
            P_theta[key] = relabel_edges(P_theta[key], new_to_id)

    return graphs, P_theta


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
