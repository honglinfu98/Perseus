# --- Get feature columns for DDM from dataset_preparation.py ---
DDM_FEATURE_COLUMNS = [
    "average_btc_base_return",
    "average_increase_percentage",
    "sum_targets_achieved",
    "rating",
    "ego_weighted_in_ratio",
    "ego_weighted_out_ratio",
    "ego_out_weights",
    "weighted_closeness_centrality",
    "weighted_betweenness_centrality",
    "weighted_pagerank",
    "ego_weighted_eff_size",
    "ego_weighted_efficiency",
    "weighted_clustering_coefficient",
    "ego_weighted_density",
]

DDINA_FEATURE_COLUMNS = [
    "average_btc_base_return",
    "average_increase_percentage",  # market
    "sum_targets_achieved",  # osn
    "rating",  # topological
    "ego_in_ratio",
    "ego_out_ratio",
    "ego_out_nodes",
    "eff_size",
    "efficiency",
    "density",
    "clustering_coeff",
    "closeness_centrality",
    "pagerank",
    "betweenness_centrality",
]
