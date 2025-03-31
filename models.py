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

class OnlineAnomalyDetector(torch.nn.Module):
    """Online anomaly detector using node embeddings.
    
    This detector maintains node embeddings and updates them based on new graph snapshots.
    Anomaly scores are computed based on the similarity between node embeddings.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.encoder = GCNEncoder(in_channels, hidden_channels, out_channels)
        self.node_embeddings = {}  # Dictionary to store node embeddings
        self.embedding_dim = out_channels
        
    def update_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor, nodes: List[int]):
        """Update node embeddings based on new graph snapshot.
        
        Args:
            x: Node features
            edge_index: Edge indices
            nodes: List of node IDs
        """
        # Get embeddings for current snapshot
        embeddings = self.encoder(x, edge_index)
        
        # Update embeddings for each node
        for i, node_id in enumerate(nodes):
            self.node_embeddings[node_id] = embeddings[i].detach()
            
    def compute_edge_score(self, src: int, dst: int) -> float:
        """Compute anomaly score for an edge based on node embeddings.
        
        Args:
            src: Source node ID
            dst: Destination node ID
            
        Returns:
            Anomaly score (higher means more likely to be anomalous)
        """
        if src not in self.node_embeddings or dst not in self.node_embeddings:
            return 0.0
            
        # Compute cosine similarity between node embeddings
        src_emb = self.node_embeddings[src]
        dst_emb = self.node_embeddings[dst]
        
        # Normalize embeddings
        src_emb = src_emb / (torch.norm(src_emb) + 1e-8)
        dst_emb = dst_emb / (torch.norm(dst_emb) + 1e-8)
        
        # Compute similarity (higher similarity = lower anomaly score)
        similarity = torch.sum(src_emb * dst_emb)
        
        # Convert to anomaly score (1 - similarity)
        return (1 - similarity).item()
    
    def state_dict(self):
        """Get the state dictionary of the model."""
        return {
            'encoder': self.encoder.state_dict(),
            'node_embeddings': {k: v.cpu() for k, v in self.node_embeddings.items()}
        }
    
    def load_state_dict(self, state_dict):
        """Load the state dictionary of the model."""
        self.encoder.load_state_dict(state_dict['encoder'])
        self.node_embeddings = {k: v.to(self.encoder.conv1.weight.device) 
                              for k, v in state_dict['node_embeddings'].items()} 