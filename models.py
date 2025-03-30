import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import Dict, List, Tuple, Optional

class GCNEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        """Graph Convolutional Network encoder.
        
        Args:
            in_channels: Number of input features
            hidden_channels: Number of hidden features
            out_channels: Number of output features
        """
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node feature matrix
            edge_index: Graph connectivity in COO format
            
        Returns:
            Node embeddings
        """
        x = self.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class TemporalGNN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        """Temporal GNN model for anomaly detection.
        
        Args:
            in_channels: Number of input features
            hidden_channels: Number of hidden features
            out_channels: Number of output features
        """
        super(TemporalGNN, self).__init__()
        self.encoder = GCNEncoder(in_channels, hidden_channels, out_channels)
        self.gru = nn.GRU(out_channels, hidden_channels, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(2 * out_channels + hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )
        
    def forward(self, data_list: List[torch.Tensor], edge_index_list: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass over a sequence of graph snapshots.
        
        Args:
            data_list: List of node feature matrices for each time step
            edge_index_list: List of edge indices for each time step
            
        Returns:
            Predicted edge scores for the next time step
        """
        # Encode each snapshot
        embeddings = []
        for x, edge_index in zip(data_list, edge_index_list):
            h = self.encoder(x, edge_index)
            # Take mean over nodes for graph-level representation
            h = h.mean(dim=0)  # [out_channels]
            embeddings.append(h)
            
        # Stack embeddings and process with GRU
        embeddings = torch.stack(embeddings)  # [T, out_channels]
        gru_out, _ = self.gru(embeddings.unsqueeze(0))  # [1, T, hidden_channels]
        
        # Use last hidden state for prediction
        last_hidden = gru_out[0, -1]  # [hidden_channels]
        
        # Get embeddings for the last snapshot
        last_embeddings = self.encoder(data_list[-1], edge_index_list[-1])  # [N, out_channels]
        
        # For each possible edge, concatenate node embeddings and predict score
        num_nodes = data_list[-1].size(0)
        edge_scores = []
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    # Concatenate node embeddings with GRU hidden state
                    node_pair = torch.cat([
                        last_embeddings[i],  # [out_channels]
                        last_embeddings[j],  # [out_channels]
                        last_hidden  # [hidden_channels]
                    ])
                    score = self.decoder(node_pair)
                    edge_scores.append(score)
                    
        return torch.stack(edge_scores)

class OnlineAnomalyDetector:
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        """Online anomaly detector using node embeddings.
        
        Args:
            in_channels: Number of input features
            hidden_channels: Number of hidden features
            out_channels: Number of output features
        """
        self.encoder = GCNEncoder(in_channels, hidden_channels, out_channels)
        self.node_embeddings: Dict[int, torch.Tensor] = {}
        self.alpha = 0.5  # blending factor for embedding updates
        
    def update_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor, 
                         node_indices: List[int]) -> None:
        """Update node embeddings for a new snapshot.
        
        Args:
            x: Node feature matrix
            edge_index: Graph connectivity
            node_indices: List of node indices in this snapshot
        """
        # Get new embeddings
        new_embeddings = self.encoder(x, edge_index)
        
        # Update embeddings for each node
        for i, node_idx in enumerate(node_indices):
            if node_idx not in self.node_embeddings:
                # New node - initialize with its embedding
                self.node_embeddings[node_idx] = new_embeddings[i]
            else:
                # Existing node - blend with new embedding
                self.node_embeddings[node_idx] = (
                    (1 - self.alpha) * self.node_embeddings[node_idx] +
                    self.alpha * new_embeddings[i]
                )
                
    def compute_edge_score(self, u: int, v: int) -> float:
        """Compute anomaly score for an edge between nodes u and v.
        
        Args:
            u: Source node index
            v: Target node index
            
        Returns:
            Anomaly score (higher means more anomalous)
        """
        if u not in self.node_embeddings or v not in self.node_embeddings:
            return 1.0  # New edge involving new node is considered anomalous
            
        # Compute cosine similarity between embeddings
        h_u = self.node_embeddings[u]
        h_v = self.node_embeddings[v]
        sim = F.cosine_similarity(h_u.unsqueeze(0), h_v.unsqueeze(0))
        
        # Convert to anomaly score (1 - similarity)
        return 1 - sim.item() 