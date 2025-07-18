import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_max_pool, GATv2Conv
from typing import Optional


class GCNNet(torch.nn.Module):
    """
    Graph Convolutional Network (GCN) with batch normalization, dropout, and skip connections.

    Args:
        num_features (int): Number of input node features.
        hidden_channels (int): Number of hidden units.
        num_classes (int): Number of output classes.
    """

    def __init__(self, num_features: int, hidden_channels: int, num_classes: int):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

        # Add batch normalization to prevent over-smoothing
        self.batch_norm1 = torch.nn.BatchNorm1d(hidden_channels)
        self.batch_norm2 = torch.nn.BatchNorm1d(num_classes)

        # Introducing residual connection
        self.skip_connection = torch.nn.Linear(num_features, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for GCNNet.

        Args:
            x (torch.Tensor): Node feature matrix [num_nodes, num_features].
            edge_index (torch.Tensor): Edge indices [2, num_edges].
            edge_weight (torch.Tensor, optional): Edge weights [num_edges].

        Returns:
            tuple: (output logits, hidden node embeddings)
        """
        # Ensure edge weights are scaled appropriately
        if edge_weight is not None:
            edge_weight = torch.abs(edge_weight)  # Make sure edge weights are positive
            edge_weight = edge_weight / torch.max(edge_weight)  # Scale to [0, 1]

        # First GCN layer with batch normalization and LeakyReLU activation
        x1 = self.conv1(x, edge_index, edge_weight=edge_weight)
        x1 = self.batch_norm1(x1)
        x1 = F.leaky_relu(x1)  # Use Leaky ReLU to allow negative slopes

        # Dropout for regularization
        x1 = F.dropout(x1, training=self.training, p=0.5)

        # Second GCN layer
        x2 = self.conv2(x1, edge_index)
        x2 = self.batch_norm2(x2)

        # Skip connection (directly from input to output)
        skip_x = self.skip_connection(x)

        # Combine the GCN output with the skip connection
        x = x2 + skip_x

        # If binary classification, use sigmoid
        # For multi-class, switch to softmax
        x = torch.sigmoid(x)  # Binary classification
        # x = torch.softmax(x, dim=1)  # Uncomment this line for multi-class classification

        return x, x1


class Net(torch.nn.Module):
    """
    Graph Attention Network (GAT) for binary classification.

    Args:
        num_features (int): Number of input node features.
        num_classes (int): Number of output classes.
        hidden_channels (int, optional): Number of hidden units. Default is 8.
    """

    def __init__(self, num_features: int, num_classes: int, hidden_channels: int = 8):
        super(Net, self).__init__()
        self.conv1 = GATConv(num_features, hidden_channels, heads=2, concat=True)
        self.conv2 = GATConv(
            hidden_channels * 2, num_classes, heads=2, concat=False
        )  # Multi-head attention reduces to the number of classes

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for GAT.

        Args:
            x (torch.Tensor): Node feature matrix [num_nodes, num_features].
            edge_index (torch.Tensor): Edge indices [2, num_edges].

        Returns:
            tuple: (output logits, hidden node embeddings)
        """
        x = F.elu(self.conv1(x, edge_index))
        embeddings = F.dropout(x, training=self.training)
        x = self.conv2(embeddings, edge_index)
        x = torch.sigmoid(x)  # Sigmoid activation for binary classification
        return x, embeddings


class GraphSAGENet(torch.nn.Module):
    """
    GraphSAGE model for binary classification.

    Args:
        num_features (int): Number of input node features.
        hidden_channels (int): Number of hidden units.
        num_classes (int): Number of output classes.
    """

    def __init__(self, num_features: int, hidden_channels: int, num_classes: int):
        super(GraphSAGENet, self).__init__()
        self.conv1 = SAGEConv(num_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for GraphSAGE.

        Args:
            x (torch.Tensor): Node feature matrix [num_nodes, num_features].
            edge_index (torch.Tensor): Edge indices [2, num_edges].
            edge_weight (torch.Tensor, optional): Edge weights [num_edges].

        Returns:
            tuple: (output logits, hidden node embeddings)
        """
        x = F.relu(self.conv1(x, edge_index))
        embeddings = F.dropout(x, training=self.training)
        x = self.conv2(embeddings, edge_index)
        x = torch.sigmoid(x)  # Sigmoid for binary

        return x, embeddings


class MultiGAT(torch.nn.Module):
    """
    Multi-graph GATv2 model with context fusion for node classification.

    Args:
        in_channels (int): Number of input node features.
        hidden_channels (int): Number of hidden units.
        heads (int, optional): Number of attention heads. Default is 8.
    """

    def __init__(self, in_channels: int, hidden_channels: int, heads: int = 8):
        super().__init__()
        # two-layer GATv2 encoder supporting edge features (edge_dim=1)
        self.conv1 = GATv2Conv(
            in_channels, hidden_channels, heads=heads, concat=True, edge_dim=1
        )
        # next layer reduces to hidden_channels (heads=1) for pooling
        self.conv2 = GATv2Conv(
            hidden_channels * heads, hidden_channels, heads=1, concat=False, edge_dim=1
        )
        # MLP on fused graph embeddings
        self.graph_mlp = torch.nn.Linear(hidden_channels, hidden_channels)
        # combine node + context
        self.node_mlp = torch.nn.Linear(hidden_channels * 2, hidden_channels)
        # final classifier → one logit per node
        self.out_lin = torch.nn.Linear(hidden_channels, 1)

    def forward(self, data_list: list) -> list:
        """
        Forward pass for MultiGAT.

        Args:
            data_list (list): List of torch_geometric.data.Data objects, each representing a graph.

        Returns:
            list: List of logits for each graph's nodes.
        """
        node_embs, graph_embs = [], []
        # encode & pool each graph
        for data in data_list:
            # prepare edge_attr for GATv2Conv
            edge_attr = getattr(data, "edge_weight", None)
            if edge_attr is not None:
                edge_attr = edge_attr.unsqueeze(-1)

            # first attention layer + activation
            h = self.conv1(x=data.x, edge_index=data.edge_index, edge_attr=edge_attr)
            h = F.elu(h)

            # second attention layer + activation
            h = self.conv2(x=h, edge_index=data.edge_index, edge_attr=edge_attr)
            h = F.elu(h)

            node_embs.append(h)
            # single-graph batch index
            batch_idx = data.x.new_zeros(h.size(0), dtype=torch.long)
            g_emb = global_max_pool(h, batch_idx)  # [1, hidden]
            graph_embs.append(g_emb)

        # # fuse all graph embeddings into one context
        # G = torch.cat(graph_embs, dim=0).max(dim=0)  # [hidden]
        # G = F.relu(self.graph_mlp(G))  # [hidden]

        # fuse all graph embeddings into one context
        G = torch.cat(graph_embs, dim=0).max(dim=0).values  # [hidden]
        G = F.relu(self.graph_mlp(G))  # [hidden]

        # inject context into nodes and classify
        logits_list = []
        for h in node_embs:
            n = h.size(0)
            G_rep = G.unsqueeze(0).expand(n, -1)  # [n, hidden]
            H_cat = torch.cat([h, G_rep], dim=1)  # [n, hidden*2]
            h2 = F.relu(self.node_mlp(H_cat))  # [n, hidden]
            logits = self.out_lin(h2).view(-1)  # [n]
            logits_list.append(logits)

        return logits_list


class MultiGraphSAGE(torch.nn.Module):
    """
    Multi-graph GraphSAGE model with context fusion for node classification.

    Args:
        in_channels (int): Number of input node features.
        hidden_channels (int): Number of hidden units.
    """

    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        # shared two-layer GraphSAGE encoder
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        # MLP on fused graph embeddings
        self.graph_mlp = torch.nn.Linear(hidden_channels, hidden_channels)
        # combine node + context
        self.node_mlp = torch.nn.Linear(hidden_channels * 2, hidden_channels)
        # final classifier → one logit per node
        self.out_lin = torch.nn.Linear(hidden_channels, 1)

    def forward(self, data_list: list) -> list:
        """
        Forward pass for MultiGraphSAGE.

        Args:
            data_list (list): List of torch_geometric.data.Data objects, each representing a graph.

        Returns:
            list: List of logits for each graph's nodes.
        """
        node_embs, graph_embs = [], []
        # encode & pool each graph
        for data in data_list:
            h = self.conv1(data.x, data.edge_index)
            h = F.relu(h)
            h = self.conv2(h, data.edge_index)
            h = F.relu(h)
            node_embs.append(h)
            # single-graph batch index
            batch_idx = data.x.new_zeros(h.size(0), dtype=torch.long)
            g_emb = global_max_pool(h, batch_idx)  # [1, hidden]
            graph_embs.append(g_emb)

        # # fuse all graph embeddings into one context
        # G = torch.cat(graph_embs, dim=0).max(dim=0)  # [hidden]
        # G = F.relu(self.graph_mlp(G))  # [hidden]

        # fuse all graph embeddings into one context
        G = torch.cat(graph_embs, dim=0).max(dim=0).values  # [hidden]
        G = F.relu(self.graph_mlp(G))  # [hidden]

        # inject context into nodes and classify
        logits_list = []
        for h in node_embs:
            n = h.size(0)
            G_rep = G.unsqueeze(0).expand(n, -1)  # [n, hidden]
            H_cat = torch.cat([h, G_rep], dim=1)  # [n, hidden*2]
            h2 = F.relu(self.node_mlp(H_cat))  # [n, hidden]
            logits = self.out_lin(h2).view(-1)  # [n]
            logits_list.append(logits)

        return logits_list
