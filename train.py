import torch
import torch.optim as optim
from torch_geometric.data import Data
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE

from data_loader import EllipticDataLoader
from models import TemporalGNN, OnlineAnomalyDetector

def train_temporal_gnn(model: TemporalGNN, train_data: List[Data], 
                      num_epochs: int = 10, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """Train the temporal GNN model.
    
    Args:
        model: TemporalGNN model instance
        train_data: List of training snapshots
        num_epochs: Number of training epochs
        device: Device to train on
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        # Process each time step
        for t in range(len(train_data) - 1):
            # Get current and next snapshot
            current_data = train_data[t].to(device)
            next_data = train_data[t + 1].to(device)
            
            # Prepare input sequence (last 5 snapshots)
            start_t = max(0, t - 4)
            data_list = [train_data[i].x.to(device) for i in range(start_t, t + 1)]
            edge_index_list = [train_data[i].edge_index.to(device) for i in range(start_t, t + 1)]
            
            # Forward pass
            edge_scores = model(data_list, edge_index_list)
            
            # Create labels (1 for edges that exist in next snapshot)
            next_edges = next_data.edge_index.t()
            labels = torch.zeros(edge_scores.size(0), device=device)
            for edge in next_edges:
                # Find corresponding score index
                score_idx = edge[0] * next_data.x.size(0) + edge[1]
                labels[score_idx] = 1
                
            # Compute loss
            loss = criterion(edge_scores, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

def train_online_detector(detector: OnlineAnomalyDetector, train_data: List[Data], 
                         device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """Train the online anomaly detector.
    
    Args:
        detector: OnlineAnomalyDetector instance
        train_data: List of training snapshots
        device: Device to train on
    """
    detector.encoder = detector.encoder.to(device)
    
    # Process each snapshot
    for data in tqdm(train_data, desc="Training online detector"):
        data = data.to(device)
        # Update embeddings for this snapshot
        detector.update_embeddings(data.x, data.edge_index, 
                                 data.nodes.tolist() if hasattr(data, 'nodes') else 
                                 list(range(data.x.size(0))))

def evaluate_model(model: TemporalGNN, test_data: List[Data], 
                  threshold: float = 0.5,
                  device: str = "cuda" if torch.cuda.is_available() else "cpu") -> Dict[str, float]:
    """Evaluate the temporal GNN model.
    
    Args:
        model: Trained TemporalGNN model
        test_data: List of test snapshots
        threshold: Threshold for anomaly detection
        device: Device to evaluate on
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for t in range(len(test_data) - 1):
            # Get current and next snapshot
            current_data = test_data[t].to(device)
            next_data = test_data[t + 1].to(device)
            
            # Prepare input sequence
            start_t = max(0, t - 4)
            data_list = [test_data[i].x.to(device) for i in range(start_t, t + 1)]
            edge_index_list = [test_data[i].edge_index.to(device) for i in range(start_t, t + 1)]
            
            # Get predictions
            edge_scores = model(data_list, edge_index_list)
            
            # Get true edges
            next_edges = next_data.edge_index.t()
            
            # Create labels (1 for illicit edges)
            labels = torch.zeros(edge_scores.size(0), device=device)
            for edge in next_edges:
                score_idx = edge[0] * next_data.x.size(0) + edge[1]
                # Check if either endpoint is illicit
                if next_data.y[edge[0]] == 1 or next_data.y[edge[1]] == 1:
                    labels[score_idx] = 1
                    
            all_scores.extend(edge_scores.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    predictions = (all_scores > threshold).astype(int)
    precision = precision_score(all_labels, predictions)
    recall = recall_score(all_labels, predictions)
    f1 = f1_score(all_labels, predictions)
    auc = roc_auc_score(all_labels, all_scores)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }

def evaluate_online_detector(detector: OnlineAnomalyDetector, test_data: List[Data],
                           threshold: float = 0.5) -> Dict[str, float]:
    """Evaluate the online anomaly detector.
    
    Args:
        detector: Trained OnlineAnomalyDetector
        test_data: List of test snapshots
        threshold: Threshold for anomaly detection
        
    Returns:
        Dictionary of evaluation metrics
    """
    all_scores = []
    all_labels = []
    
    for t in range(len(test_data) - 1):
        current_data = test_data[t]
        next_data = test_data[t + 1]
        
        # Update embeddings for current snapshot
        detector.update_embeddings(current_data.x, current_data.edge_index,
                                 current_data.nodes.tolist() if hasattr(current_data, 'nodes') else 
                                 list(range(current_data.x.size(0))))
        
        # Score edges in next snapshot
        next_edges = next_data.edge_index.t()
        for edge in next_edges:
            score = detector.compute_edge_score(edge[0].item(), edge[1].item())
            # Check if either endpoint is illicit
            label = 1 if next_data.y[edge[0]] == 1 or next_data.y[edge[1]] == 1 else 0
            
            all_scores.append(score)
            all_labels.append(label)
    
    # Convert to numpy arrays
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    predictions = (all_scores > threshold).astype(int)
    precision = precision_score(all_labels, predictions)
    recall = recall_score(all_labels, predictions)
    f1 = f1_score(all_labels, predictions)
    auc = roc_auc_score(all_labels, all_scores)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }

def visualize_anomalies(model: TemporalGNN, data: Data, threshold: float = 0.5,
                       device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """Visualize detected anomalies in a graph snapshot.
    
    Args:
        model: Trained TemporalGNN model
        data: Graph snapshot to visualize
        threshold: Threshold for anomaly detection
        device: Device to use
    """
    # Create NetworkX graph
    G = nx.DiGraph()
    
    # Add nodes
    for i in range(data.x.size(0)):
        G.add_node(i, label=data.y[i].item())
        
    # Add edges
    edge_index = data.edge_index.t()
    for edge in edge_index:
        G.add_edge(edge[0].item(), edge[1].item())
    
    # Get node embeddings
    with torch.no_grad():
        embeddings = model.encoder(data.x.to(device), data.edge_index.to(device))
        embeddings = embeddings.cpu().numpy()
    
    # Project embeddings to 2D
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(15, 5))
    
    # Graph structure
    plt.subplot(121)
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color=['red' if G.nodes[n]['label'] == 1 else 'blue' 
                                              for n in G.nodes()])
    nx.draw_networkx_edges(G, pos)
    plt.title("Graph Structure (Red=Illicit)")
    
    # Embedding space
    plt.subplot(122)
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
               c=['red' if G.nodes[n]['label'] == 1 else 'blue' 
                  for n in G.nodes()])
    plt.title("Node Embeddings (Red=Illicit)")
    
    plt.tight_layout()
    plt.show()

def main():
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load data
    data_loader = EllipticDataLoader()
    train_data, test_data = data_loader.get_train_test_split()
    
    # Initialize models
    in_channels = data_loader.get_node_features_dim()
    hidden_channels = 64
    out_channels = 32
    
    temporal_gnn = TemporalGNN(in_channels, hidden_channels, out_channels)
    online_detector = OnlineAnomalyDetector(in_channels, hidden_channels, out_channels)
    
    # Train models
    print("Training Temporal GNN...")
    train_temporal_gnn(temporal_gnn, train_data)
    
    print("\nTraining Online Detector...")
    train_online_detector(online_detector, train_data)
    
    # Evaluate models
    print("\nEvaluating Temporal GNN...")
    temporal_metrics = evaluate_model(temporal_gnn, test_data)
    print(f"Temporal GNN Metrics:")
    for metric, value in temporal_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nEvaluating Online Detector...")
    online_metrics = evaluate_online_detector(online_detector, test_data)
    print(f"Online Detector Metrics:")
    for metric, value in online_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Visualize anomalies
    print("\nVisualizing anomalies...")
    visualize_anomalies(temporal_gnn, test_data[0])

if __name__ == "__main__":
    main() 