import torch
import numpy as np
from data_loader import EllipticDataLoader
from models import TemporalGNN, OnlineAnomalyDetector
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import json
import os
from torch_geometric.data import Data
from typing import Dict, List

def load_trained_models(model_path: str, data_loader: EllipticDataLoader, device: str):
    """Load trained models from disk.
    
    Args:
        model_path: Path to saved model
        data_loader: DataLoader instance
        device: Device to load models on
        
    Returns:
        Tuple of (temporal_gnn, online_detector)
    """
    in_channels = data_loader.get_node_feature_dim()
    hidden_channels = 64
    out_channels = 32
    
    # Initialize models
    temporal_gnn = TemporalGNN(in_channels, hidden_channels, out_channels).to(device)
    online_detector = OnlineAnomalyDetector(in_channels, hidden_channels, out_channels)
    
    # Load trained weights
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        temporal_gnn.load_state_dict(checkpoint['temporal_gnn'])
        online_detector.load_state_dict(checkpoint['online_detector'])
        print(f"Loaded models from {model_path}")
        print("Model metrics:")
        print(f"Temporal GNN: {checkpoint['metrics']['temporal_gnn']}")
        print(f"Online Detector: {checkpoint['metrics']['online_detector']}")
    else:
        print(f"No saved models found at {model_path}")
    
    return temporal_gnn, online_detector

def detect_anomalies(temporal_gnn: TemporalGNN, online_detector: OnlineAnomalyDetector,
                    data: Data, threshold: float = 0.5, device: str = "cuda"):
    """Detect anomalies using both models.
    
    Args:
        temporal_gnn: Trained TemporalGNN model
        online_detector: Trained OnlineAnomalyDetector
        data: Graph snapshot to analyze
        threshold: Threshold for anomaly detection
        device: Device to use
        
    Returns:
        Dictionary containing anomaly scores and predictions
    """
    temporal_gnn.eval()
    data = data.to(device)
    
    # Get predictions from temporal GNN
    with torch.no_grad():
        pred_scores = temporal_gnn([data.x], [data.edge_index])
        pred_scores = pred_scores.squeeze(-1)
    
    # Get predictions from online detector
    online_detector.update_embeddings(data.x, data.edge_index,
                                    data.nodes.tolist() if hasattr(data, 'nodes') else 
                                    list(range(data.x.size(0))))
    
    edge_scores = []
    for edge in data.edge_index.t():
        score = online_detector.compute_edge_score(edge[0].item(), edge[1].item())
        edge_scores.append(score)
    
    # Combine predictions
    edge_scores = torch.tensor(edge_scores, device=device)
    combined_scores = (pred_scores + edge_scores) / 2
    
    # Get predictions
    predictions = (combined_scores > threshold).astype(int)
    
    return {
        'scores': combined_scores.cpu().numpy(),
        'predictions': predictions,
        'true_labels': data.y.cpu().numpy()
    }

def plot_roc_curve(scores: np.ndarray, labels: np.ndarray):
    """Plot ROC curve for anomaly detection.
    
    Args:
        scores: Anomaly scores
        labels: True labels
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

def analyze_anomalies(results: dict, data_loader: EllipticDataLoader):
    """Analyze detected anomalies.
    
    Args:
        results: Dictionary containing detection results
        data_loader: DataLoader instance
    """
    scores = results['scores']
    predictions = results['predictions']
    true_labels = results['true_labels']
    
    # Calculate statistics
    num_anomalies = np.sum(predictions)
    num_true_anomalies = np.sum(true_labels)
    anomaly_ratio = num_anomalies / len(predictions)
    
    print("\nAnomaly Detection Analysis:")
    print(f"Total edges analyzed: {len(predictions)}")
    print(f"Detected anomalies: {num_anomalies}")
    print(f"True anomalies: {num_true_anomalies}")
    print(f"Anomaly ratio: {anomaly_ratio:.2%}")
    
    # Plot ROC curve
    plot_roc_curve(scores, true_labels)
    
    # Save results
    results_dict = {
        'total_edges': len(predictions),
        'detected_anomalies': int(num_anomalies),
        'true_anomalies': int(num_true_anomalies),
        'anomaly_ratio': float(anomaly_ratio),
        'scores': scores.tolist(),
        'predictions': predictions.tolist(),
        'true_labels': true_labels.tolist()
    }
    
    with open('anomaly_results.json', 'w') as f:
        json.dump(results_dict, f)
    
    print("\nResults saved to anomaly_results.json")

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    data_loader = EllipticDataLoader(sample_size=0.05)
    
    # Load trained models
    model_path = 'trained_models.pt'
    temporal_gnn, online_detector = load_trained_models(model_path, data_loader, device)
    
    # Get test data
    _, test_snapshots = data_loader.get_train_test_split()
    
    if not test_snapshots:
        print("No test data available")
        return
    
    # Analyze each test snapshot
    for i, snapshot in enumerate(test_snapshots):
        print(f"\nAnalyzing snapshot {i+1}/{len(test_snapshots)}")
        results = detect_anomalies(temporal_gnn, online_detector, snapshot, device=device)
        analyze_anomalies(results, data_loader)

if __name__ == "__main__":
    main() 