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
import torch.nn.functional as F

from data_loader import EllipticDataLoader
from models import TemporalGNN, OnlineAnomalyDetector

def train_temporal_gnn(model: TemporalGNN, train_data: List[Data], optimizer: torch.optim.Optimizer, num_epochs: int, device: torch.device):
    """Train the temporal GNN model.
    
    Args:
        model: The temporal GNN model
        train_data: List of training snapshots
        optimizer: Optimizer for training
        num_epochs: Number of training epochs
        device: Device to train on (CPU/GPU)
    """
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        # Add progress bar for snapshots
        pbar = tqdm(range(len(train_data) - 1), desc=f'Epoch {epoch+1}/{num_epochs}')
        for t in pbar:
            # Get current and next snapshot
            current_snapshot = train_data[t].to(device)
            next_snapshot = train_data[t + 1].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            # Pass node features and edge indices separately
            pred_scores = model([current_snapshot.x], [current_snapshot.edge_index])
            # Squeeze the predictions to match label dimensions
            pred_scores = pred_scores.squeeze(-1)
            
            # Compute loss (binary cross-entropy with logits)
            # We'll use the presence/absence of edges in the next snapshot as labels
            next_edges = next_snapshot.edge_index.t()
            num_nodes = current_snapshot.x.size(0)
            
            # Create a set of existing edges for faster lookup
            existing_edges = set((edge[0].item(), edge[1].item()) for edge in next_edges)
            
            # Instead of creating labels for all possible edges, only create labels for:
            # 1. Existing edges in next snapshot
            # 2. A random sample of non-existent edges
            num_negative_samples = min(len(existing_edges) * 2, num_nodes * (num_nodes - 1) // 2 - len(existing_edges))
            negative_edges = set()
            
            # Sample negative edges
            while len(negative_edges) < num_negative_samples:
                i = torch.randint(0, num_nodes, (1,)).item()
                j = torch.randint(0, num_nodes, (1,)).item()
                if i != j and (i, j) not in existing_edges:
                    negative_edges.add((i, j))
            
            # Combine positive and negative edges
            all_edges = list(existing_edges) + list(negative_edges)
            
            # Create labels only for sampled edges
            edge_labels = torch.zeros(len(all_edges), device=device)
            edge_labels[:len(existing_edges)] = 1  # Positive edges
            
            # Get predictions only for sampled edges
            edge_indices = []
            for i, j in all_edges:
                # Calculate edge index in the upper triangular matrix
                if i < j:
                    edge_idx = i * (num_nodes - 1) + j - 1
                else:
                    edge_idx = j * (num_nodes - 1) + i - 1
                # Only add index if it's within bounds
                if edge_idx < pred_scores.size(0):
                    edge_indices.append(edge_idx)
            
            # If we have no valid indices, skip this snapshot
            if not edge_indices:
                print(f"Warning: No valid edge indices found for snapshot {t}")
                continue
                
            edge_indices = torch.tensor(edge_indices, device=device)
            pred_scores = pred_scores[edge_indices]
            
            # Ensure predictions and labels have the same size
            if pred_scores.size(0) != edge_labels.size(0):
                print(f"Warning: Size mismatch - pred_scores: {pred_scores.size(0)}, edge_labels: {edge_labels.size(0)}")
                # Trim to the smaller size
                min_size = min(pred_scores.size(0), edge_labels.size(0))
                pred_scores = pred_scores[:min_size]
                edge_labels = edge_labels[:min_size]
            
            loss = F.binary_cross_entropy_with_logits(pred_scores, edge_labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        avg_loss = total_loss / (len(train_data) - 1)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

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
            
            # Forward pass
            pred_scores = model([current_data.x], [current_data.edge_index])
            pred_scores = pred_scores.squeeze(-1)
            
            # Get true edges
            next_edges = next_data.edge_index.t()
            num_nodes = current_data.x.size(0)
            
            # Create labels (1 for illicit edges)
            # Only consider edges that are within bounds
            valid_edges = []
            valid_labels = []
            
            for edge in next_edges:
                src, dst = edge[0].item(), edge[1].item()
                # Calculate edge index in the upper triangular matrix
                if src < dst:
                    edge_idx = src * (num_nodes - 1) + dst - 1
                else:
                    edge_idx = dst * (num_nodes - 1) + src - 1
                
                # Only add if the edge index is within bounds
                if edge_idx < pred_scores.size(0):
                    valid_edges.append(edge_idx)
                    # Check if either endpoint is illicit
                    label = 1 if next_data.y[src] == 1 or next_data.y[dst] == 1 else 0
                    valid_labels.append(label)
            
            if valid_edges:  # Only process if we have valid edges
                valid_edges = torch.tensor(valid_edges, device=device)
                valid_scores = pred_scores[valid_edges]
                
                all_scores.extend(valid_scores.cpu().numpy())
                all_labels.extend(valid_labels)
    
    # Convert to numpy arrays
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    if len(all_scores) == 0 or len(all_labels) == 0:
        print("Warning: No valid edges found for evaluation")
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auc": 0.0
        }
    
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
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data with sampling (use 5% of the data for testing)
    data_loader = EllipticDataLoader(sample_size=0.05)  # Reduced from 0.1 to 0.05
    
    # Get train/test split
    train_snapshots, test_snapshots = data_loader.get_train_test_split()
    
    # Initialize model
    in_channels = data_loader.get_node_feature_dim()
    hidden_channels = 64
    out_channels = 32
    
    model = TemporalGNN(in_channels, hidden_channels, out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop
    num_epochs = 3  # Reduced from 5 to 3
    print("Starting training...")
    train_temporal_gnn(model, train_snapshots, optimizer, num_epochs, device)
    
    # Evaluation
    print("\nEvaluating model...")
    metrics = evaluate_model(model, test_snapshots, threshold=0.5, device=device)
    print(f"Test Metrics: {metrics}")
    
    # Initialize and train online detector
    print("\nTraining online detector...")
    online_detector = OnlineAnomalyDetector(in_channels, hidden_channels, out_channels)
    train_online_detector(online_detector, train_snapshots)
    
    # Evaluate online detector
    print("\nEvaluating online detector...")
    online_metrics = evaluate_online_detector(online_detector, test_snapshots)
    print(f"Online Detector Metrics: {online_metrics}")
    
    # Save trained models
    print("\nSaving trained models...")
    checkpoint = {
        'temporal_gnn': model.state_dict(),
        'online_detector': online_detector.state_dict(),
        'metrics': {
            'temporal_gnn': {k: float(v) for k, v in metrics.items()},
            'online_detector': {k: float(v) for k, v in online_metrics.items()}
        }
    }
    torch.save(checkpoint, 'trained_models.pt', _use_new_zipfile_serialization=True)
    print("Models saved to trained_models.pt")
    
    # Visualize anomalies in a test snapshot
    if len(test_snapshots) > 0:
        print("\nVisualizing anomalies...")
        visualize_anomalies(model, test_snapshots[0], device=device)

if __name__ == "__main__":
    main() 