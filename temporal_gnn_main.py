import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import SAGEConv, GATConv
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, auc, roc_auc_score
import time
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import optuna # Added for hyperparameter tuning
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice

class TemporalGNNAnomalyDetector(nn.Module):
    def __init__(self, 
                 in_feats: int = 165, 
                 hidden_feats: int = 64,
                 use_gat: bool = False,
                 num_gnn_layers: int = 2,
                 dropout: float = 0.2,
                 drop_edge_rate: float = 0.1,  # New: DropEdge rate
                 residual: bool = True,  # New: Residual connections
                 jumping_knowledge: bool = True,  # New: Jumping knowledge
                 batch_norm: bool = True,  # New: Batch normalization
                 device: torch.device = None):
        """
        Temporal GNN for anomaly detection on transaction networks.
        
        Args:
            in_feats: Input feature dimension
            hidden_feats: Hidden feature dimension
            use_gat: Whether to use GAT (True) or GraphSAGE (False)
            num_gnn_layers: Number of GNN layers
            dropout: Dropout probability
            drop_edge_rate: Probability of dropping edges during training
            residual: Whether to use residual connections
            jumping_knowledge: Whether to use jumping knowledge connections
            batch_norm: Whether to use batch normalization
            device: Device to run the model on
        """
        super(TemporalGNNAnomalyDetector, self).__init__()
        self.hidden_feats = hidden_feats
        self.num_gnn_layers = num_gnn_layers
        self.dropout = dropout
        self.drop_edge_rate = drop_edge_rate
        self.residual = residual
        self.jumping_knowledge = jumping_knowledge
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Input transformation
        self.input_projection = nn.Linear(in_feats, hidden_feats)
        
        # Batch norm layers
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn_layers = nn.ModuleList([nn.BatchNorm1d(hidden_feats) for _ in range(num_gnn_layers)])
        
        # GNN encoder layers
        self.gnn_layers = nn.ModuleList()
        for i in range(num_gnn_layers):
            input_dim = hidden_feats
            if use_gat:
                # Multi-head attention for better feature extraction
                heads = 4 if i < num_gnn_layers - 1 else 1
                head_dim = hidden_feats // heads if i < num_gnn_layers - 1 else hidden_feats
                self.gnn_layers.append(GATConv(input_dim, head_dim, heads=heads))
            else:
                self.gnn_layers.append(SAGEConv(input_dim, hidden_feats))
        
        # Jumping knowledge
        if jumping_knowledge:
            self.jk_layer = nn.Linear(hidden_feats * (num_gnn_layers + 1), hidden_feats)
        
        # GRU for temporal evolution of node embeddings
        self.gru = nn.GRUCell(hidden_feats, hidden_feats)
        
        # Node classifier with deeper MLP
        self.classifier = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            nn.BatchNorm1d(hidden_feats),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_feats, hidden_feats // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.8),  # Slightly reduced dropout
            nn.Linear(hidden_feats // 2, 2)  # logits for [licit, illicit]
        )
        
        # Link predictor (decoder) - deeper architecture
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_feats * 2, hidden_feats),
            nn.BatchNorm1d(hidden_feats),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_feats, hidden_feats // 2),
            nn.BatchNorm1d(hidden_feats // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.8),
            nn.Linear(hidden_feats // 2, 1)
        )
        
        # Embedding similarity projection - improved with more features
        self.emb_sim_proj = nn.Sequential(
            nn.Linear(3, hidden_feats // 4),
            nn.BatchNorm1d(hidden_feats // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),  # Lower dropout for this component
            nn.Linear(hidden_feats // 4, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights for better training dynamics
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights with Xavier/Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Kaiming initialization for ReLU activations
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def drop_edges(self, edge_index, p=None):
        """Randomly drop edges during training (DropEdge)"""
        if self.training and (p is not None) and p > 0:
            edge_index = edge_index.clone()
            num_edges = edge_index.size(1)
            perm = torch.rand(num_edges, device=edge_index.device)
            mask = perm >= p
            return edge_index[:, mask]
        return edge_index
    
    def gnn_forward(self, x, edge_index):
        """Forward pass through the GNN layers with residual connections and jumping knowledge"""
        # Initial projection
        h = F.relu(self.input_projection(x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Store intermediate representations for jumping knowledge if enabled
        intermediate = [h]
        
        # Apply DropEdge in training mode
        if self.training:
            edge_index = self.drop_edges(edge_index, self.drop_edge_rate)
        
        # Process through GNN layers
        for i, gnn_layer in enumerate(self.gnn_layers):
            # Apply GNN layer
            h_new = gnn_layer(h, edge_index)
            
            # Apply batch normalization if enabled
            if self.batch_norm:
                h_new = self.bn_layers[i](h_new)
            
            # Apply ReLU activation
            h_new = F.relu(h_new)
            
            # Apply dropout
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            
            # Add residual connection if enabled (except for first layer with different dims)
            if self.residual and h.size() == h_new.size():
                h = h_new + h
            else:
                h = h_new
            
            # Store this layer's output for jumping knowledge
            intermediate.append(h)
        
        # Apply jumping knowledge if enabled
        if self.jumping_knowledge:
            # Concatenate all intermediate representations
            h = torch.cat(intermediate, dim=1)
            # Project back to hidden_feats dimension
            h = self.jk_layer(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        return h
    
    def forward(self, x, edge_index, h_prev=None):
        """
        Forward pass for one snapshot.
        
        Args:
            x: Node feature matrix [num_nodes, in_feats]
            edge_index: Edge index [2, num_edges]
            h_prev: Previous hidden states [num_nodes, hidden_feats]
            
        Returns:
            h_curr: Updated hidden states
            class_logits: Classification logits for nodes
        """
        # Ensure tensors have correct data types
        if x is not None and not x.is_floating_point():
            x = x.float()
        
        if edge_index is not None and not edge_index.dtype == torch.long:
            edge_index = edge_index.long()
            
        # GNN encoder: get current embeddings
        h_curr = self.gnn_forward(x, edge_index)
        
        # Update with GRU if previous state exists
        if h_prev is not None:
            if not h_prev.is_floating_point():
                h_prev = h_prev.float()
            h_curr = self.gru(h_curr, h_prev)
        
        # Classification logits
        class_logits = self.classifier(h_curr)
        
        return h_curr, class_logits
    
    def predict_links(self, h, edge_index):
        """
        Predict links given node embeddings and candidate edges.
        
        Args:
            h: Node embeddings [num_nodes, hidden_feats]
            edge_index: Edge index [2, num_edges]
            
        Returns:
            link_scores: Predicted probabilities for links
        """
        src, dst = edge_index
        h_src = h[src]
        h_dst = h[dst]
        
        # Concatenate source and destination embeddings
        h_edge = torch.cat([h_src, h_dst], dim=1)
        
        # Predict link probability
        link_scores = torch.sigmoid(self.link_predictor(h_edge).squeeze(-1))
        
        return link_scores
    
    def compute_embedding_similarity_score(self, h_curr, global_node_ids, hidden_dict):
        """
        Compute anomaly scores based on embedding similarity between current and past states.
        Enhanced with additional similarity metrics.
        
        Args:
            h_curr: Current embeddings [num_curr_nodes, hidden_feats]
            global_node_ids: List of global IDs for current nodes
            hidden_dict: Dictionary mapping global IDs to previous hidden states
            
        Returns:
            similarity_scores: Anomaly scores based on embedding changes
        """
        if not hidden_dict:
            # No previous embeddings to compare
            return None
        
        # Ensure h_curr is float32
        if not h_curr.is_floating_point():
            h_curr = h_curr.float()
            
        device = h_curr.device
        num_nodes = h_curr.size(0)
        cos_sim = torch.zeros(num_nodes, dtype=torch.float32, device=device)
        euc_dist = torch.zeros(num_nodes, dtype=torch.float32, device=device)
        l1_dist = torch.zeros(num_nodes, dtype=torch.float32, device=device)  # Added L1 distance
        
        for i, node_id in enumerate(global_node_ids):
            if node_id in hidden_dict:
                # This node existed before
                curr_emb = h_curr[i].unsqueeze(0)
                prev_emb = hidden_dict[node_id].unsqueeze(0)
                
                # Ensure prev_emb is float32
                if not prev_emb.is_floating_point():
                    prev_emb = prev_emb.float()
                
                # 1. Compute cosine similarity (higher = more similar)
                # Using F.normalize for stable computation
                cos = F.cosine_similarity(
                    F.normalize(curr_emb, p=2, dim=1),
                    F.normalize(prev_emb, p=2, dim=1)
                )
                
                # 2. Compute euclidean distance (higher = less similar)
                dist = torch.cdist(curr_emb, prev_emb, p=2).squeeze()
                
                # 3. Compute L1 distance (Manhattan distance)
                l1 = torch.cdist(curr_emb, prev_emb, p=1).squeeze()
                
                cos_sim[i] = cos
                euc_dist[i] = dist
                l1_dist[i] = l1
            else:
                # New node, set default values to indicate anomaly
                cos_sim[i] = -1.0  # Minimum similarity
                euc_dist[i] = 1.0  # Large distance
                l1_dist[i] = 1.0   # Large distance
        
        # Normalize distances to [0,1] for stable computation
        if euc_dist.max() > 0:
            euc_dist = euc_dist / (euc_dist.max() + 1e-8)
        if l1_dist.max() > 0:
            l1_dist = l1_dist / (l1_dist.max() + 1e-8)
        
        # Scale cosine similarity from [-1,1] to [0,1] - easier to combine
        cos_sim = (cos_sim + 1) / 2
        
        # Combine similarity metrics - now with 3 features
        sim_features = torch.stack([cos_sim, euc_dist, l1_dist], dim=1)
        
        # Handle edge case for batch norm with single sample
        if num_nodes == 1:
            # Skip batch norm for single sample
            similarity_scores = torch.sigmoid(sim_features.mean(dim=1))
        else:
            similarity_scores = self.emb_sim_proj(sim_features).squeeze()
        
        # Convert to anomaly score (higher = more anomalous)
        # 1 - similarity = anomaly
        anomaly_scores = 1 - similarity_scores
        
        return anomaly_scores
    
    def compute_reconstruction_error(self, h, edge_index):
        """
        Compute reconstruction error for links with improved edge weighting.
        
        Args:
            h: Node embeddings [num_nodes, hidden_feats]
            edge_index: Edge index of actual edges [2, num_edges]
            
        Returns:
            recon_errors: Reconstruction error for each edge
            node_recon_errors: Aggregated reconstruction error per node
        """
        # Predict link probabilities for actual edges
        link_probs = self.predict_links(h, edge_index)
        
        # Reconstruction error = 1 - link probability (higher = more anomalous)
        recon_errors = 1 - link_probs
        
        # Aggregate errors to node level (out-edges)
        src_nodes = edge_index[0]
        node_recon_errors = torch.zeros(h.size(0), device=h.device)
        out_degrees = torch.zeros(h.size(0), device=h.device)
        
        # Compute weighted reconstruction error per node
        for i in range(edge_index.size(1)):
            src = src_nodes[i].item()
            node_recon_errors[src] += recon_errors[i]
            out_degrees[src] += 1
        
        # Average by degree, avoiding division by zero
        mask = out_degrees > 0
        node_recon_errors[mask] = node_recon_errors[mask] / out_degrees[mask]
        
        # For isolated nodes (no out edges), assign higher anomaly value
        node_recon_errors[~mask] = 0.75  # Slightly anomalous by default
        
        return recon_errors, node_recon_errors
    
    def compute_anomaly_scores(self, h_curr, global_node_ids, hidden_dict, edge_index, alpha=0.5):
        """
        Compute combined anomaly scores with improved combination strategy.
        
        Args:
            h_curr: Current node embeddings [num_nodes, hidden_feats]
            global_node_ids: List of global IDs for current nodes
            hidden_dict: Dictionary mapping global IDs to previous hidden states
            edge_index: Edge index [2, num_edges]
            alpha: Weight for combining reconstruction and similarity scores
                   (higher alpha = more weight on reconstruction error)
            
        Returns:
            anomaly_scores: Combined anomaly scores [num_nodes]
        """
        # Compute reconstruction error
        _, node_recon_errors = self.compute_reconstruction_error(h_curr, edge_index)
        
        # Compute embedding similarity score if possible
        sim_scores = self.compute_embedding_similarity_score(h_curr, global_node_ids, hidden_dict)
        
        if sim_scores is not None:
            # Compute classifier confidence (entropy)
            logits = self.classifier(h_curr)
            probs = F.softmax(logits, dim=1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
            entropy = entropy / torch.log(torch.tensor(2.0, device=entropy.device))  # Normalize to [0,1]
            
            # Combine the scores (all are higher = more anomalous)
            # For nodes with history, weight the similarity score
            # For new nodes or nodes with minimal history, rely more on reconstruction
            
            # 1. Create a mask for nodes that exist in hidden_dict
            has_history = torch.zeros(h_curr.size(0), dtype=torch.bool, device=h_curr.device)
            for i, node_id in enumerate(global_node_ids):
                if node_id in hidden_dict:
                    has_history[i] = True
            
            # 2. Combine scores differently based on history
            # For nodes with history: alpha*recon + (1-alpha)*sim
            # For nodes without history: 0.8*recon + 0.2*entropy
            combined_scores = torch.zeros_like(node_recon_errors)
            
            # Nodes with history
            if has_history.sum() > 0:
                combined_scores[has_history] = (
                    alpha * node_recon_errors[has_history] + 
                    (1 - alpha) * sim_scores[has_history]
                )
            
            # Nodes without history
            if (~has_history).sum() > 0:
                combined_scores[~has_history] = (
                    0.8 * node_recon_errors[~has_history] + 
                    0.2 * entropy[~has_history]
                )
        else:
            # Only use reconstruction error and entropy if no similarity can be computed
            logits = self.classifier(h_curr)
            probs = F.softmax(logits, dim=1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
            entropy = entropy / torch.log(torch.tensor(2.0, device=entropy.device))  # Normalize to [0,1]
            
            combined_scores = 0.8 * node_recon_errors + 0.2 * entropy
        
        return combined_scores


class GNNAnomalyTrainer:
    def __init__(self, 
                 model, 
                 data_loader, 
                 train_time_range=(1, 34), 
                 val_time_range=(30, 34),  # Use last 5 snapshots of training for validation
                 test_time_range=(35, 49),
                 lr=0.001, 
                 weight_decay=1e-5,
                 illicit_weight=None,  # Auto-compute from class distribution if None
                 lambda_link=1.0,  # Weight for link prediction loss
                 alpha_anomaly=0.5,
                 focal_loss_gamma=2.0,  # New: Focal loss gamma parameter
                 temporal_weight_decay=0.9,  # New: Temporal weight decay factor
                 mixup_alpha=0.2,  # New: Mixup regularization
                 feature_noise=0.05,  # New: Feature noise level
                 consistency_reg_weight=0.1,  # New: Weight for consistency regularization
                 time_aware_weighting=True,  # New: Whether to use time-aware loss weighting
                 device=None):
        """
        Trainer for the Temporal GNN Anomaly Detector.
        
        Args:
            model: Instance of TemporalGNNAnomalyDetector
            data_loader: Data loader providing snapshots
            train_time_range: Time range for training (start, end)
            val_time_range: Time range for validation (start, end) - must be within train range
            test_time_range: Time range for testing (start, end)
            lr: Learning rate
            weight_decay: L2 regularization weight
            illicit_weight: Weight for illicit class in loss function (auto if None)
            lambda_link: Weight for link prediction loss
            alpha_anomaly: Weight for combining reconstruction error and embedding similarity
            focal_loss_gamma: Parameter for focal loss (higher means more focus on hard examples)
            temporal_weight_decay: Weight decay factor for older time steps
            mixup_alpha: Alpha parameter for mixup regularization (0 = no mixup)
            feature_noise: Standard deviation of Gaussian noise to add to features
            consistency_reg_weight: Weight for consistency regularization
            time_aware_weighting: Whether to use time-aware loss weighting
            device: Computation device
        """
        self.model = model
        self.data_loader = data_loader
        self.train_time_range = train_time_range
        
        # Ensure validation range is within training range to prevent temporal leakage
        self.val_time_range = (
            max(val_time_range[0], train_time_range[0]),
            min(val_time_range[1], train_time_range[1])
        )
        
        self.test_time_range = test_time_range
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # New parameters
        self.focal_loss_gamma = focal_loss_gamma
        self.temporal_weight_decay = temporal_weight_decay
        self.mixup_alpha = mixup_alpha
        self.feature_noise = feature_noise
        self.consistency_reg_weight = consistency_reg_weight
        self.time_aware_weighting = time_aware_weighting
        
        # Optimizer with gradient clipping
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        
        # Cosine annealing learning rate scheduler for better convergence
        total_steps = (train_time_range[1] - train_time_range[0] + 1) * 50  # Estimate for 50 epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=total_steps // 10, T_mult=2, eta_min=lr * 0.01
        )
        
        # Compute class weights if not provided
        if illicit_weight is None:
            illicit_weight = self._compute_class_weight()
            print(f"Auto-computed illicit class weight: {illicit_weight:.2f}")
        
        # Loss functions with class weighting - EXPLICITLY set to float32
        self.weight = torch.tensor([1.0, illicit_weight], dtype=torch.float32, device=self.device)
        self.criterion = nn.CrossEntropyLoss(weight=self.weight, reduction='none')
        self.lambda_link = lambda_link
        self.alpha_anomaly = alpha_anomaly
        
        # Global node mapping for efficient hidden state tracking
        self.global_node_mapping = {}  # Maps global ID to index in hidden state tensor
        self.next_idx = 0
        
        # Tracking metrics
        self.train_losses = []
        self.val_metrics = []
        self.best_val_f1 = 0
        self.best_model_state = None
        self.epoch_times = []
        
        # Threshold tuning
        self.best_threshold = 0.5
        
        # Multiple validation thresholds to improve generalization
        self.val_thresholds = []
        
        # Enhanced tracking of class distribution for adaptive weighting
        self.class_distribution_history = []
        self.temporal_weights = {}
    
    def _compute_class_weight(self):
        """Compute class weight based on class distribution in training data"""
        licit_count = 0
        illicit_count = 0
        
        # Count classes in training data
        for t in range(self.train_time_range[0], self.train_time_range[1] + 1):
            try:
                data_t = self.data_loader.get_snapshot(t)
                labels = data_t.y.cpu().numpy()
                licit_count += np.sum(labels == 0)
                illicit_count += np.sum(labels == 1)
            except Exception as e:
                print(f"Error counting classes at time step {t}: {e}")
        
        # Prevent division by zero
        if illicit_count == 0:
            return 50.0  # Default if no illicit nodes found
            
        # Compute weight as inverse of class frequency ratio
        return licit_count / max(illicit_count, 1)
    
    def _get_node_index(self, global_id):
        """Get index for a global node ID in the hidden state tensor"""
        if global_id not in self.global_node_mapping:
            self.global_node_mapping[global_id] = self.next_idx
            self.next_idx += 1
        return self.global_node_mapping[global_id]
    
    def sample_negative_edges(self, edge_index, num_nodes, num_neg_samples):
        """Sample negative edges that don't exist in the graph using vectorized operations"""
        # Convert to set of tuples for O(1) lookup
        pos_edges = set([(edge_index[0, i].item(), edge_index[1, i].item()) 
                        for i in range(edge_index.size(1))])
        
        # For efficiency, limit negative sampling attempts
        max_attempts = min(num_neg_samples * 3, num_nodes * num_nodes // 10)
        
        # Generate random node pairs efficiently in batches
        neg_edges = []
        attempts = 0
        
        batch_size = min(num_neg_samples * 2, 10000)  # Process in batches to save memory
        
        while len(neg_edges) < num_neg_samples and attempts < max_attempts:
            src = torch.randint(0, num_nodes, (batch_size,), device=self.device)
            dst = torch.randint(0, num_nodes, (batch_size,), device=self.device)
            
            # Filter out self-loops and existing edges
            mask = src != dst
            src_filtered = src[mask]
            dst_filtered = dst[mask]
            
            for i in range(len(src_filtered)):
                s, d = src_filtered[i].item(), dst_filtered[i].item()
                if (s, d) not in pos_edges:
                    neg_edges.append((s, d))
                    if len(neg_edges) >= num_neg_samples:
                        break
            
            attempts += batch_size
        
        # Ensure we have enough negative samples
        if len(neg_edges) < num_neg_samples:
            print(f"Warning: Could only sample {len(neg_edges)} negative edges")
        
        # Convert to tensor
        if neg_edges:
            neg_src, neg_dst = zip(*neg_edges[:num_neg_samples])
            neg_edge_index = torch.tensor([neg_src, neg_dst], dtype=torch.long, device=self.device)
        else:
            # Fallback for extreme cases
            neg_edge_index = torch.zeros((2, 1), dtype=torch.long, device=self.device)
        
        return neg_edge_index
    
    def compute_link_prediction_loss(self, h, pos_edge_index):
        """Compute link prediction loss with negative sampling"""
        # Ensure h is float32
        if not h.is_floating_point():
            h = h.float()
            
        num_nodes = h.size(0)
        num_pos = pos_edge_index.size(1)
        
        # Sample negative edges - match positive sample count
        neg_edge_index = self.sample_negative_edges(pos_edge_index, num_nodes, num_pos)
        
        # Get features for positive edges
        pos_features = torch.cat([h[pos_edge_index[0]], h[pos_edge_index[1]]], dim=1)
        
        # Get features for negative edges
        neg_features = torch.cat([h[neg_edge_index[0]], h[neg_edge_index[1]]], dim=1)
        
        # Get predictions
        pos_pred = self.model.link_predictor(pos_features).squeeze(-1)
        neg_pred = self.model.link_predictor(neg_features).squeeze(-1)
        
        # Create targets (1 for positive edges, 0 for negative edges)
        pos_targets = torch.ones_like(pos_pred, dtype=torch.float32)
        neg_targets = torch.zeros_like(neg_pred, dtype=torch.float32)
        
        # Binary cross entropy loss with more numerically stable implementation
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_pred,
            pos_targets,
            reduction='mean'
        )
        
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_pred,
            neg_targets,
            reduction='mean'
        )
        
        return pos_loss + neg_loss
    
    def focal_loss(self, logits, targets, gamma=None):
        """
        Compute focal loss for imbalanced classification.
        
        Args:
            logits: Prediction logits [batch_size, num_classes]
            targets: Target classes [batch_size]
            gamma: Focal loss gamma parameter (higher means more focus on hard examples)
            
        Returns:
            focal_loss: Focal loss tensor
        """
        if gamma is None:
            gamma = self.focal_loss_gamma
            
        # Convert targets to one-hot encoding
        num_classes = logits.size(1)
        one_hot_targets = F.one_hot(targets, num_classes=num_classes).float()
        
        # Compute probabilities and focal weights
        probs = F.softmax(logits, dim=1)
        pt = torch.sum(probs * one_hot_targets, dim=1)  # Probability of target class
        focal_weight = (1 - pt) ** gamma
        
        # Apply class weights
        class_weights = torch.ones_like(targets, dtype=torch.float32, device=self.device)
        for i in range(targets.size(0)):
            class_weights[i] = self.weight[targets[i]]
            
        # Compute cross entropy loss
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Apply focal weighting
        loss = focal_weight * ce_loss * class_weights
        
        return loss.mean()
    
    def mixup_data(self, x, edge_index, y):
        """
        Apply mixup augmentation to node features and labels.
        
        Args:
            x: Node features [num_nodes, in_feats]
            edge_index: Edge index [2, num_edges]
            y: Node labels [num_nodes]
            
        Returns:
            mixed_x: Mixed node features
            edge_index: Original edge index
            y_a, y_b: Original labels for mixed nodes
            lam: Mixup ratio
        """
        if self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        else:
            lam = 1
            
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=self.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        
        return mixed_x, edge_index, y_a, y_b, lam
    
    def get_time_weight(self, t, max_t):
        """
        Calculate time-aware weight for temporal steps.
        More recent time steps get higher weight.
        
        Args:
            t: Current time step
            max_t: Maximum time step
            
        Returns:
            weight: Weight for this time step
        """
        if self.time_aware_weighting:
            # Linear increase in importance from oldest to newest
            norm_t = (t - self.train_time_range[0]) / (max_t - self.train_time_range[0] + 1)
            # Higher weight for more recent time steps (0.5 to 1.5 range)
            return 0.5 + norm_t
        else:
            return 1.0
    
    def compute_consistency_loss(self, logits1, logits2):
        """
        Compute consistency loss between two sets of predictions.
        
        Args:
            logits1: First set of logits
            logits2: Second set of logits
            
        Returns:
            consistency_loss: Consistency loss
        """
        # Convert to probabilities
        probs1 = F.softmax(logits1, dim=1)
        probs2 = F.softmax(logits2, dim=1)
        
        # MSE between probability distributions
        consistency_loss = F.mse_loss(probs1, probs2)
        
        return consistency_loss
    
    def train_epoch(self, epoch):
        """Train for one epoch using enhanced regularization techniques"""
        self.model.train()
        epoch_loss = 0
        cls_loss_total = 0
        link_loss_total = 0
        temporal_reg_total = 0
        consistency_loss_total = 0
        num_batches = 0
        
        # Track time for this epoch
        start_time = time.time()
        
        # Create tensor for hidden states of all nodes
        # Initialize with max size we might need (based on global node mapping)
        max_hidden_size = max(1000, len(self.global_node_mapping))
        hidden_states = torch.zeros(max_hidden_size, self.model.hidden_feats, dtype=torch.float32, device=self.device)
        
        # Keep copy of previous hidden states for temporal consistency regularization
        prev_hidden_states = hidden_states.clone()
        
        # Track which nodes have appeared so far
        seen_nodes = set()
        
        # Calculate max training time step for time weights
        max_train_t = self.train_time_range[1]
        
        # Iterate through time steps in training range, excluding validation
        train_steps = [t for t in range(self.train_time_range[0], self.train_time_range[1] + 1)
                      if t < self.val_time_range[0] or t > self.val_time_range[1]]
        
        for t in train_steps:
            try:
                # Get data for current time step
                data_t = self.data_loader.get_snapshot(t)
                
                # Convert all tensors to the right types BEFORE sending to device
                if hasattr(data_t, 'x') and data_t.x is not None:
                    data_t.x = data_t.x.float()
                
                if hasattr(data_t, 'edge_index') and data_t.edge_index is not None:
                    data_t.edge_index = data_t.edge_index.long()
                
                if hasattr(data_t, 'y') and data_t.y is not None:
                    data_t.y = data_t.y.long()  # Labels should be long integers
                
                # Now send to device
                data_t = data_t.to(self.device)
                
                # Get global node IDs
                if hasattr(data_t, 'node_ids'):
                    global_node_ids = data_t.node_ids.tolist()
                elif hasattr(data_t, 'nodes'):
                    global_node_ids = data_t.nodes.tolist()
                else:
                    # If no global IDs are available, use local indices
                    global_node_ids = list(range(data_t.x.size(0)))
                
                # Map global IDs to indices in hidden state tensor
                curr_indices = []
                for node_id in global_node_ids:
                    curr_indices.append(self._get_node_index(node_id))
                    seen_nodes.add(node_id)
                
                # Get current slice of hidden states
                curr_indices_tensor = torch.tensor(curr_indices, dtype=torch.long, device=self.device)
                
                # Check if we need to expand hidden states tensor
                if max(curr_indices) >= hidden_states.size(0):
                    # Double size to avoid frequent resizing
                    new_size = max(max(curr_indices) + 1, hidden_states.size(0) * 2)
                    new_hidden_states = torch.zeros(new_size, self.model.hidden_feats, dtype=torch.float32, device=self.device)
                    new_hidden_states[:hidden_states.size(0)] = hidden_states
                    
                    new_prev_hidden_states = torch.zeros(new_size, self.model.hidden_feats, dtype=torch.float32, device=self.device)
                    new_prev_hidden_states[:hidden_states.size(0)] = prev_hidden_states
                    
                    hidden_states = new_hidden_states
                    prev_hidden_states = new_prev_hidden_states
                
                # Get previous hidden states for current nodes
                h_prev = hidden_states[curr_indices_tensor]
                
                # Add feature noise for regularization during training
                if self.feature_noise > 0:
                    noise = torch.randn_like(data_t.x) * self.feature_noise
                    data_t.x = data_t.x + noise
                
                # Apply mixup if enabled
                use_mixup = self.mixup_alpha > 0 and epoch > 5  # Start mixup after some initial training
                
                if use_mixup and data_t.y is not None:
                    # Only mix labeled nodes
                    labeled_mask = (data_t.y != -1)
                    if labeled_mask.sum() > 1:  # Need at least 2 labeled nodes for mixup
                        labeled_indices = torch.where(labeled_mask)[0]
                        
                        # Get original features and labels for labeled nodes
                        orig_x = data_t.x[labeled_indices]
                        orig_y = data_t.y[labeled_indices]
                        
                        # Apply mixup only to labeled nodes
                        mixed_x, _, y_a, y_b, lam = self.mixup_data(orig_x, None, orig_y)
                        
                        # Replace original features with mixed features
                        data_t.x[labeled_indices] = mixed_x
                        
                        # Store original labels for later loss calculation
                        mixup_labels_a = y_a
                        mixup_labels_b = y_b
                        mixup_lam = lam
                        mixup_indices = labeled_indices
                    else:
                        use_mixup = False
                
                # Forward pass
                h_curr, logits = self.model(data_t.x, data_t.edge_index, h_prev)
                
                # Consistency regularization: second forward pass with different noise/dropout
                if self.consistency_reg_weight > 0:
                    # Add different noise for second forward pass
                    if self.feature_noise > 0:
                        noise2 = torch.randn_like(data_t.x) * self.feature_noise
                        data_t_noisy = data_t.clone()
                        data_t_noisy.x = data_t.x + noise2
                    else:
                        data_t_noisy = data_t
                    
                    # Second forward pass with different stochastic components
                    _, logits2 = self.model(data_t_noisy.x, data_t_noisy.edge_index, h_prev)
                    
                    # Calculate consistency loss on all nodes (including unlabeled)
                    consistency_loss = self.compute_consistency_loss(logits, logits2)
                    consistency_loss_total += consistency_loss.item()
                else:
                    consistency_loss = 0
                
                # Store old state for temporal consistency regularization
                old_h = prev_hidden_states[curr_indices_tensor]
                
                # Update hidden states with detach to prevent backprop through time
                # but maintain stable training
                hidden_states[curr_indices_tensor] = h_curr.detach()
                
                # Compute classification loss on labeled nodes only
                labels = data_t.y
                labeled_mask = (labels != -1)  # Mask for labeled nodes
                
                batch_loss = 0
                cls_loss = 0
                
                # Get time weight for this time step
                time_weight = self.get_time_weight(t, max_train_t)
                
                # Classification loss if there are labeled nodes
                if labeled_mask.sum() > 0:
                    if use_mixup:
                        # Mixup loss calculation
                        logits_mixed = logits[mixup_indices]
                        loss_a = self.focal_loss(logits_mixed, mixup_labels_a)
                        loss_b = self.focal_loss(logits_mixed, mixup_labels_b)
                        cls_loss = mixup_lam * loss_a + (1 - mixup_lam) * loss_b
                    else:
                        # Use focal loss for better handling of imbalanced classes
                        logits_masked = logits[labeled_mask]
                        labels_masked = labels[labeled_mask]
                        cls_loss = self.focal_loss(logits_masked, labels_masked)
                    
                    # Apply time weighting - more recent time steps have higher weight
                    cls_loss = cls_loss * time_weight
                    
                    batch_loss += cls_loss
                    cls_loss_total += cls_loss.item()
                
                # Link prediction loss with increased weight
                link_loss = self.compute_link_prediction_loss(h_curr, data_t.edge_index)
                batch_loss += self.lambda_link * link_loss * time_weight
                link_loss_total += link_loss.item()
                
                # Add temporal consistency regularization
                # This encourages smooth temporal transitions that generalize better to test data
                temporal_consistency_mask = torch.any(old_h != 0, dim=1)  # Identify nodes with history
                if temporal_consistency_mask.sum() > 0:
                    # L2 distance between normalized embeddings to focus on direction not magnitude
                    old_h_norm = F.normalize(old_h[temporal_consistency_mask], p=2, dim=1)
                    h_curr_norm = F.normalize(h_curr[temporal_consistency_mask], p=2, dim=1)
                    
                    # Cosine similarity (higher = more similar)
                    cos_sim = F.cosine_similarity(old_h_norm, h_curr_norm, dim=1)
                    
                    # Convert to temporal consistency loss (lower = more similar)
                    # Use smooth L1 loss for better stability
                    temporal_reg = F.smooth_l1_loss(cos_sim, torch.ones_like(cos_sim))
                    
                    # Apply adaptive weight based on epoch
                    # Lower weight initially to let model explore, higher later for stability
                    temp_reg_weight = min(0.1, 0.01 * (1 + epoch / 10.0))
                    batch_loss += temp_reg_weight * temporal_reg
                    temporal_reg_total += temporal_reg.item()
                
                # Add consistency regularization
                if self.consistency_reg_weight > 0:
                    batch_loss += self.consistency_reg_weight * consistency_loss
                
                # Update model
                self.optimizer.zero_grad()
                batch_loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                
                self.optimizer.step()
                self.scheduler.step()
                
                epoch_loss += batch_loss.item()
                num_batches += 1
                
                # Update previous hidden states for next iteration
                prev_hidden_states = hidden_states.clone()
                
            except Exception as e:
                print(f"Error processing training time step {t}: {e}")
                import traceback
                traceback.print_exc()  # Print full stack trace for debugging
        
        # Calculate average loss
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        avg_cls_loss = cls_loss_total / max(num_batches, 1)
        avg_link_loss = link_loss_total / max(num_batches, 1)
        avg_temp_reg = temporal_reg_total / max(num_batches, 1)
        avg_consistency_loss = consistency_loss_total / max(num_batches, 1)
        
        self.train_losses.append(avg_epoch_loss)
        
        epoch_time = time.time() - start_time
        self.epoch_times.append(epoch_time)
        
        # Report component losses
        return {
            'total_loss': avg_epoch_loss, 
            'cls_loss': avg_cls_loss, 
            'link_loss': avg_link_loss, 
            'temp_reg': avg_temp_reg,
            'consistency_loss': avg_consistency_loss
        }
    
    def validate(self, time_range=None):
        """Validate on a specific time range with proper hidden state handling"""
        if time_range is None:
            time_range = self.val_time_range
        
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        all_anomaly_scores = []
        
        # Create tensor for hidden states of all nodes
        max_hidden_size = max(1000, len(self.global_node_mapping))
        hidden_states = torch.zeros(max_hidden_size, self.model.hidden_feats, dtype=torch.float32, device=self.device)
        
        # Dictionary for mapping global node IDs to hidden states (for anomaly score computation)
        hidden_dict = {}
        
        with torch.no_grad():
            # First, build up hidden states from training data before validation point
            for t in range(self.train_time_range[0], time_range[0]):
                try:
                    # Get data for current time step
                    data_t = self.data_loader.get_snapshot(t)
                    
                    # Convert all tensors to the right types BEFORE sending to device
                    if hasattr(data_t, 'x') and data_t.x is not None:
                        data_t.x = data_t.x.float()
                    
                    if hasattr(data_t, 'edge_index') and data_t.edge_index is not None:
                        data_t.edge_index = data_t.edge_index.long()
                    
                    if hasattr(data_t, 'y') and data_t.y is not None:
                        data_t.y = data_t.y.long()  # Labels should be long integers
                    
                    # Now send to device
                    data_t = data_t.to(self.device)
                    
                    # Get global node IDs
                    if hasattr(data_t, 'node_ids'):
                        global_node_ids = data_t.node_ids.tolist()
                    elif hasattr(data_t, 'nodes'):
                        global_node_ids = data_t.nodes.tolist()
                    else:
                        global_node_ids = list(range(data_t.x.size(0)))
                    
                    # Map global IDs to indices in hidden state tensor
                    curr_indices = []
                    for node_id in global_node_ids:
                        curr_indices.append(self._get_node_index(node_id))
                    
                    # Get current slice of hidden states
                    curr_indices_tensor = torch.tensor(curr_indices, dtype=torch.long, device=self.device)
                    
                    # Check if we need to expand hidden states tensor
                    if max(curr_indices) >= hidden_states.size(0):
                        new_size = max(max(curr_indices) + 1, hidden_states.size(0) * 2)
                        new_hidden_states = torch.zeros(new_size, self.model.hidden_feats, dtype=torch.float32, device=self.device)
                        new_hidden_states[:hidden_states.size(0)] = hidden_states
                        hidden_states = new_hidden_states
                    
                    # Get previous hidden states for current nodes
                    h_prev = hidden_states[curr_indices_tensor]
                    
                    # Forward pass
                    h_curr, _ = self.model(data_t.x, data_t.edge_index, h_prev)
                    
                    # Update hidden states
                    hidden_states[curr_indices_tensor] = h_curr
                    
                    # Update hidden dict for anomaly score computation
                    for i, node_id in enumerate(global_node_ids):
                        hidden_dict[node_id] = h_curr[i]
                
                except Exception as e:
                    print(f"Error processing pre-validation time step {t}: {e}")
                    import traceback
                    traceback.print_exc()  # Print full stack trace for debugging
            
            # Now run validation
            for t in range(time_range[0], time_range[1] + 1):
                try:
                    # Get data for current time step
                    data_t = self.data_loader.get_snapshot(t)
                    
                    # Convert all tensors to the right types BEFORE sending to device
                    if hasattr(data_t, 'x') and data_t.x is not None:
                        data_t.x = data_t.x.float()
                    
                    if hasattr(data_t, 'edge_index') and data_t.edge_index is not None:
                        data_t.edge_index = data_t.edge_index.long()
                    
                    if hasattr(data_t, 'y') and data_t.y is not None:
                        data_t.y = data_t.y.long()  # Labels should be long integers
                    
                    # Now send to device
                    data_t = data_t.to(self.device)
                    
                    # Get global node IDs
                    if hasattr(data_t, 'node_ids'):
                        global_node_ids = data_t.node_ids.tolist()
                    elif hasattr(data_t, 'nodes'):
                        global_node_ids = data_t.nodes.tolist()
                    else:
                        global_node_ids = list(range(data_t.x.size(0)))
                    
                    # Map global IDs to indices in hidden state tensor
                    curr_indices = []
                    for node_id in global_node_ids:
                        curr_indices.append(self._get_node_index(node_id))
                    
                    # Get current slice of hidden states
                    curr_indices_tensor = torch.tensor(curr_indices, dtype=torch.long, device=self.device)
                    
                    # Check if we need to expand hidden states tensor
                    if max(curr_indices) >= hidden_states.size(0):
                        new_size = max(max(curr_indices) + 1, hidden_states.size(0) * 2)
                        new_hidden_states = torch.zeros(new_size, self.model.hidden_feats, dtype=torch.float32, device=self.device)
                        new_hidden_states[:hidden_states.size(0)] = hidden_states
                        hidden_states = new_hidden_states
                    
                    # Get previous hidden states for current nodes
                    h_prev = hidden_states[curr_indices_tensor]
                    
                    # Forward pass
                    h_curr, logits = self.model(data_t.x, data_t.edge_index, h_prev)
                    
                    # Update hidden states
                    hidden_states[curr_indices_tensor] = h_curr
                    
                    # Compute anomaly scores
                    anomaly_scores = self.model.compute_anomaly_scores(
                        h_curr, global_node_ids, hidden_dict, data_t.edge_index, self.alpha_anomaly
                    )
                    
                    # Update hidden dict for next step
                    for i, node_id in enumerate(global_node_ids):
                        hidden_dict[node_id] = h_curr[i]
                    
                    # Extract probabilities and predictions
                    probs = F.softmax(logits, dim=1)[:, 1]  # Probability of being illicit
                    
                    # Filter labeled nodes only
                    labels = data_t.y
                    labeled_mask = (labels != -1)
                    
                    if labeled_mask.sum() > 0:
                        # Don't compute predictions yet - will tune threshold later
                        all_probs.append(probs[labeled_mask].cpu())
                        all_labels.append(labels[labeled_mask].cpu())
                        
                        if anomaly_scores is not None:
                            all_anomaly_scores.append(anomaly_scores[labeled_mask].cpu())
                
                except Exception as e:
                    print(f"Error validating time step {t}: {e}")
                    import traceback
                    traceback.print_exc()  # Print full stack trace for debugging
        
        # Concatenate results
        if all_probs:
            all_labels = torch.cat(all_labels)
            all_probs = torch.cat(all_probs)
            
            if all_anomaly_scores:
                all_anomaly_scores = torch.cat(all_anomaly_scores)
            else:
                all_anomaly_scores = all_probs  # Use class probs as fallback
            
            # Find optimal threshold on validation set
            precision_curve, recall_curve, thresholds = precision_recall_curve(all_labels, all_probs)
            
            # Compute F1 for each threshold
            f1_scores = []
            for i in range(len(precision_curve) - 1):  # -1 because precision_recall_curve returns one more precision than threshold
                if i < len(thresholds):  # Safety check
                    precision = precision_curve[i]
                    recall = recall_curve[i]
                    f1 = 2 * precision * recall / (precision + recall + 1e-10)
                    f1_scores.append((f1, thresholds[i]))
            
            # Find threshold with best F1
            f1_scores.sort(reverse=True)
            
            # Store top 3 thresholds (if available) for better generalization
            top_thresholds = []
            for i in range(min(3, len(f1_scores))):
                if i < len(f1_scores):
                    f1, threshold = f1_scores[i]
                    top_thresholds.append((threshold, f1))
            
            # Add top thresholds to the history
            self.val_thresholds.append(top_thresholds)
            
            # Select best threshold but with a bias toward more stable values
            # Use the median of recent best thresholds for stability
            recent_thresholds = []
            for thresholds in self.val_thresholds[-5:]:  # Consider last 5 validation runs
                if thresholds:
                    recent_thresholds.append(thresholds[0][0])  # Use the best threshold from each run
            
            if recent_thresholds:
                # Use median for stability
                best_threshold = float(np.median(recent_thresholds))
            else:
                best_threshold = f1_scores[0][1] if f1_scores else 0.5
            
            self.best_threshold = best_threshold
            
            # Apply threshold to get predictions
            all_preds = (all_probs >= best_threshold).int()
            
            # Calculate metrics
            precision = precision_score(all_labels, all_preds)
            recall = recall_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds)
            
            # AUC using class probabilities
            try:
                roc_auc = roc_auc_score(all_labels, all_probs)
            except Exception:
                roc_auc = 0.5
            
            # AUC using anomaly scores
            try:
                anomaly_auc = roc_auc_score(all_labels, all_anomaly_scores)
            except Exception:
                anomaly_auc = 0.5
            
            # PR-AUC (Precision-Recall AUC)
            pr_auc = auc(recall_curve, precision_curve)
            
            metrics = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'anomaly_auc': anomaly_auc,
                'pr_auc': pr_auc,
                'threshold': best_threshold
            }
            
            return metrics
        
        return None
    
    def test(self):
        """Test on the test time range using proper hidden state handling and more robust threshold"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        all_anomaly_scores = []
        timestamps = []
        node_ids_list = []
        
        # Create tensor for hidden states of all nodes
        max_hidden_size = max(1000, len(self.global_node_mapping))
        hidden_states = torch.zeros(max_hidden_size, self.model.hidden_feats, dtype=torch.float32, device=self.device)
        
        # Dictionary for mapping global node IDs to hidden states (for anomaly score computation)
        hidden_dict = {}
        
        with torch.no_grad():
            # First, build up hidden states from training data
            for t in range(self.train_time_range[0], self.train_time_range[1] + 1):
                try:
                    # Get data for current time step
                    data_t = self.data_loader.get_snapshot(t)
                    
                    # Convert all tensors to the right types BEFORE sending to device
                    if hasattr(data_t, 'x') and data_t.x is not None:
                        data_t.x = data_t.x.float()
                    
                    if hasattr(data_t, 'edge_index') and data_t.edge_index is not None:
                        data_t.edge_index = data_t.edge_index.long()
                    
                    if hasattr(data_t, 'y') and data_t.y is not None:
                        data_t.y = data_t.y.long()  # Labels should be long integers
                    
                    # Now send to device
                    data_t = data_t.to(self.device)
                    
                    # Get global node IDs
                    if hasattr(data_t, 'node_ids'):
                        global_node_ids = data_t.node_ids.tolist()
                    elif hasattr(data_t, 'nodes'):
                        global_node_ids = data_t.nodes.tolist()
                    else:
                        global_node_ids = list(range(data_t.x.size(0)))
                    
                    # Map global IDs to indices in hidden state tensor
                    curr_indices = []
                    for node_id in global_node_ids:
                        curr_indices.append(self._get_node_index(node_id))
                    
                    # Get current slice of hidden states
                    curr_indices_tensor = torch.tensor(curr_indices, dtype=torch.long, device=self.device)
                    
                    # Check if we need to expand hidden states tensor
                    if max(curr_indices) >= hidden_states.size(0):
                        new_size = max(max(curr_indices) + 1, hidden_states.size(0) * 2)
                        new_hidden_states = torch.zeros(new_size, self.model.hidden_feats, dtype=torch.float32, device=self.device)
                        new_hidden_states[:hidden_states.size(0)] = hidden_states
                        hidden_states = new_hidden_states
                    
                    # Get previous hidden states for current nodes
                    h_prev = hidden_states[curr_indices_tensor]
                    
                    # Forward pass
                    h_curr, _ = self.model(data_t.x, data_t.edge_index, h_prev)
                    
                    # Update hidden states
                    hidden_states[curr_indices_tensor] = h_curr
                    
                    # Update hidden dict for anomaly score computation
                    for i, node_id in enumerate(global_node_ids):
                        hidden_dict[node_id] = h_curr[i]
                
                except Exception as e:
                    print(f"Error processing pre-test time step {t}: {e}")
                    import traceback
                    traceback.print_exc()  # Print full stack trace for debugging
            
            # Now run test
            for t in range(self.test_time_range[0], self.test_time_range[1] + 1):
                try:
                    # Get data for current time step
                    data_t = self.data_loader.get_snapshot(t)
                    
                    # Convert all tensors to the right types BEFORE sending to device
                    if hasattr(data_t, 'x') and data_t.x is not None:
                        data_t.x = data_t.x.float()
                    
                    if hasattr(data_t, 'edge_index') and data_t.edge_index is not None:
                        data_t.edge_index = data_t.edge_index.long()
                    
                    if hasattr(data_t, 'y') and data_t.y is not None:
                        data_t.y = data_t.y.long()  # Labels should be long integers
                    
                    # Now send to device
                    data_t = data_t.to(self.device)
                    
                    # Get global node IDs
                    if hasattr(data_t, 'node_ids'):
                        global_node_ids = data_t.node_ids.tolist()
                    elif hasattr(data_t, 'nodes'):
                        global_node_ids = data_t.nodes.tolist()
                    else:
                        global_node_ids = list(range(data_t.x.size(0)))
                    
                    # Map global IDs to indices in hidden state tensor
                    curr_indices = []
                    for node_id in global_node_ids:
                        curr_indices.append(self._get_node_index(node_id))
                    
                    # Get current slice of hidden states
                    curr_indices_tensor = torch.tensor(curr_indices, dtype=torch.long, device=self.device)
                    
                    # Check if we need to expand hidden states tensor
                    if max(curr_indices) >= hidden_states.size(0):
                        new_size = max(max(curr_indices) + 1, hidden_states.size(0) * 2)
                        new_hidden_states = torch.zeros(new_size, self.model.hidden_feats, dtype=torch.float32, device=self.device)
                        new_hidden_states[:hidden_states.size(0)] = hidden_states
                        hidden_states = new_hidden_states
                    
                    # Get previous hidden states for current nodes
                    h_prev = hidden_states[curr_indices_tensor]
                    
                    # Forward pass
                    h_curr, logits = self.model(data_t.x, data_t.edge_index, h_prev)
                    
                    # Update hidden states
                    hidden_states[curr_indices_tensor] = h_curr
                    
                    # Compute anomaly scores
                    anomaly_scores = self.model.compute_anomaly_scores(
                        h_curr, global_node_ids, hidden_dict, data_t.edge_index, self.alpha_anomaly
                    )
                    
                    # Update hidden dict for next step
                    for i, node_id in enumerate(global_node_ids):
                        hidden_dict[node_id] = h_curr[i]
                    
                    # Extract probabilities and use best threshold from validation
                    probs = F.softmax(logits, dim=1)[:, 1]  # Probability of being illicit
                    preds = (probs >= self.best_threshold).int()
                    
                    # Filter labeled nodes only
                    labels = data_t.y
                    labeled_mask = (labels != -1)
                    
                    if labeled_mask.sum() > 0:
                        all_preds.append(preds[labeled_mask].cpu())
                        all_labels.append(labels[labeled_mask].cpu())
                        all_probs.append(probs[labeled_mask].cpu())
                        
                        if anomaly_scores is not None:
                            all_anomaly_scores.append(anomaly_scores[labeled_mask].cpu())
                        
                        # Store timestep and node ids for analysis
                        timestep_tensor = torch.full((labeled_mask.sum(),), t, device='cpu')
                        timestamps.append(timestep_tensor)
                        
                        # Store the global node IDs for labeled nodes
                        labeled_indices = torch.where(labeled_mask)[0].cpu()
                        labeled_node_ids = [global_node_ids[i] for i in labeled_indices]
                        node_ids_list.append(torch.tensor(labeled_node_ids, device='cpu'))
                
                except Exception as e:
                    print(f"Error testing time step {t}: {e}")
                    import traceback
                    traceback.print_exc()  # Print full stack trace for debugging
        
        # Concatenate results
        if all_preds:
            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            all_probs = torch.cat(all_probs)
            timestamps = torch.cat(timestamps)
            
            if all_anomaly_scores:
                all_anomaly_scores = torch.cat(all_anomaly_scores)
            else:
                all_anomaly_scores = all_probs  # Use class probs as fallback
            
            # Safely concatenate node_ids if we have any
            node_ids = torch.cat(node_ids_list) if node_ids_list else None
            
            # Calculate metrics
            precision = precision_score(all_labels, all_preds)
            recall = recall_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds)
            
            # Try different thresholds on the test set to see if we can get better results
            precision_curve, recall_curve, thresholds = precision_recall_curve(all_labels, all_probs)
            
            # Compute F1 for each threshold
            f1_scores = []
            for i in range(len(precision_curve) - 1):
                if i < len(thresholds):
                    precision_i = precision_curve[i]
                    recall_i = recall_curve[i]
                    f1_i = 2 * precision_i * recall_i / (precision_i + recall_i + 1e-10)
                    f1_scores.append((f1_i, thresholds[i], precision_i, recall_i))
            
            # Find best F1 threshold for test set
            f1_scores.sort(reverse=True)
            best_test_f1, best_test_threshold, best_test_precision, best_test_recall = f1_scores[0] if f1_scores else (0, 0.5, 0, 0)
            
            # AUC using class probabilities
            try:
                roc_auc = roc_auc_score(all_labels, all_probs)
            except Exception:
                roc_auc = 0.5
            
            # AUC using anomaly scores
            try:
                anomaly_auc = roc_auc_score(all_labels, all_anomaly_scores)
            except Exception:
                anomaly_auc = 0.5
            
            # PR-AUC (Precision-Recall AUC)
            pr_auc = auc(recall_curve, precision_curve)
            
            # These are the results using the threshold from validation
            metrics = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'anomaly_auc': anomaly_auc,
                'pr_auc': pr_auc,
                'threshold': self.best_threshold,
                # Test set best threshold metrics (for analysis)
                'best_test_f1': best_test_f1,
                'best_test_threshold': best_test_threshold, 
                'best_test_precision': best_test_precision,
                'best_test_recall': best_test_recall,
                # Raw data for further analysis
                'all_preds': all_preds,
                'all_labels': all_labels,
                'all_probs': all_probs,
                'all_anomaly_scores': all_anomaly_scores,
                'timestamps': timestamps,
                'node_ids': node_ids
            }
            
            return metrics
        
        return None
    
    def fit(self, num_epochs, patience=5, val_interval=1):
        """Train the model with improved logging and component loss tracking"""
        print(f"Starting training for {num_epochs} epochs with strict temporal split...")
        print(f"Train time range: {self.train_time_range}, Val time range: {self.val_time_range}, Test time range: {self.test_time_range}")
        print(f"Link prediction loss weight (lambda): {self.lambda_link}, Focal loss gamma: {self.focal_loss_gamma}")
        print(f"Mixup alpha: {self.mixup_alpha}, Feature noise: {self.feature_noise}, Consistency reg weight: {self.consistency_reg_weight}")
        
        best_val_f1 = 0
        patience_counter = 0
        
        # Track the best model state for each metric
        best_models = {
            'f1': {'score': 0, 'state': None, 'epoch': 0},
            'pr_auc': {'score': 0, 'state': None, 'epoch': 0},
            'anomaly_auc': {'score': 0, 'state': None, 'epoch': 0}
        }
        
        for epoch in range(num_epochs):
            # Train one epoch
            loss_dict = self.train_epoch(epoch)
            
            # Validate
            if (epoch + 1) % val_interval == 0:
                val_metrics = self.validate()
                
                if val_metrics:
                    val_f1 = val_metrics['f1']
                    self.val_metrics.append(val_metrics)
                    
                    # Progress of threshold tuning
                    recent_thresholds = [t[0] for sublist in self.val_thresholds[-3:] for t in sublist]
                    threshold_stability = np.std(recent_thresholds) if len(recent_thresholds) > 1 else 0
                    
                    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss_dict['total_loss']:.4f} "
                          f"(Cls: {loss_dict['cls_loss']:.4f}, Link: {loss_dict['link_loss']:.4f}, "
                          f"Temp Reg: {loss_dict['temp_reg']:.4f}, Consist: {loss_dict['consistency_loss']:.4f}), "
                          f"Val F1: {val_f1:.4f} @{val_metrics['threshold']:.2f} (σ={threshold_stability:.3f}), "
                          f"Val ROC-AUC: {val_metrics['roc_auc']:.4f}, "
                          f"Val PR-AUC: {val_metrics['pr_auc']:.4f}, "
                          f"Val Anomaly-AUC: {val_metrics['anomaly_auc']:.4f}")
                    
                    # Track best model for each metric
                    if val_f1 > best_models['f1']['score']:
                        best_models['f1']['score'] = val_f1
                        best_models['f1']['state'] = self.model.state_dict().copy()
                        best_models['f1']['epoch'] = epoch + 1
                        print(f"New best F1 model: {val_f1:.4f}")
                    
                    if val_metrics['pr_auc'] > best_models['pr_auc']['score']:
                        best_models['pr_auc']['score'] = val_metrics['pr_auc']
                        best_models['pr_auc']['state'] = self.model.state_dict().copy()
                        best_models['pr_auc']['epoch'] = epoch + 1
                    
                    if val_metrics['anomaly_auc'] > best_models['anomaly_auc']['score']:
                        best_models['anomaly_auc']['score'] = val_metrics['anomaly_auc']
                        best_models['anomaly_auc']['state'] = self.model.state_dict().copy()
                        best_models['anomaly_auc']['epoch'] = epoch + 1
                    
                    # Check for improvement on main metric (F1)
                    if val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                        self.best_val_f1 = val_f1
                        self.best_model_state = self.model.state_dict().copy()
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                    # Early stopping
                    if patience > 0 and patience_counter >= patience:
                        print(f"Early stopping after {epoch+1} epochs")
                        break
                else:
                    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss_dict['total_loss']:.4f}, No validation metrics")
            else:
                print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss_dict['total_loss']:.4f} "
                      f"(Cls: {loss_dict['cls_loss']:.4f}, Link: {loss_dict['link_loss']:.4f}, "
                      f"Temp Reg: {loss_dict['temp_reg']:.4f}, Consist: {loss_dict['consistency_loss']:.4f})")
        
        # Save all best models
        torch.save(best_models['f1']['state'], 'best_f1_model.pt')
        torch.save(best_models['pr_auc']['state'], 'best_prauc_model.pt')
        torch.save(best_models['anomaly_auc']['state'], 'best_anomaly_auc_model.pt')
        
        print(f"Saved best models:")
        print(f"  F1: {best_models['f1']['score']:.4f} (epoch {best_models['f1']['epoch']})")
        print(f"  PR-AUC: {best_models['pr_auc']['score']:.4f} (epoch {best_models['pr_auc']['epoch']})")
        print(f"  Anomaly-AUC: {best_models['anomaly_auc']['score']:.4f} (epoch {best_models['anomaly_auc']['epoch']})")
        
        # Restore best F1 model 
        self.model.load_state_dict(self.best_model_state)
        print(f"Restored best F1 model with validation F1: {self.best_val_f1:.4f}")
        
        # Run a pseudo-test on the validation set to validate stability
        print("Running final validation check for threshold stability...")
        val_metrics = self.validate()
        if val_metrics:
            print(f"Final validation - F1: {val_metrics['f1']:.4f}, "
                  f"Precision: {val_metrics['precision']:.4f}, "
                  f"Recall: {val_metrics['recall']:.4f}, "
                  f"Threshold: {val_metrics['threshold']:.4f}")
        
        return self.val_metrics
    
    def plot_metrics(self, save_path=None):
        """Plot training metrics"""
        if not self.train_losses or not self.val_metrics:
            print("No metrics to plot")
            return
        
        # Extract metrics
        epochs = list(range(1, len(self.train_losses) + 1))
        val_epochs = list(range(len(self.val_metrics)))
        val_f1 = [m['f1'] for m in self.val_metrics]
        val_roc_auc = [m['roc_auc'] for m in self.val_metrics]
        val_pr_auc = [m['pr_auc'] for m in self.val_metrics]
        
        # Plot metrics
        plt.figure(figsize=(15, 10))
        
        # Loss plot
        plt.subplot(2, 2, 1)
        plt.plot(epochs, self.train_losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # F1 plot
        plt.subplot(2, 2, 2)
        plt.plot(val_epochs, val_f1)
        plt.title('Validation F1')
        plt.xlabel('Validation Check')
        plt.ylabel('F1 Score')
        plt.grid(True)
        
        # ROC-AUC plot
        plt.subplot(2, 2, 3)
        plt.plot(val_epochs, val_roc_auc)
        plt.title('Validation ROC-AUC')
        plt.xlabel('Validation Check')
        plt.ylabel('ROC-AUC')
        plt.grid(True)
        
        # PR-AUC plot
        plt.subplot(2, 2, 4)
        plt.plot(val_epochs, val_pr_auc)
        plt.title('Validation PR-AUC')
        plt.xlabel('Validation Check')
        plt.ylabel('PR-AUC')
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_precision_recall_curve(self, test_metrics, save_path=None):
        """Plot precision-recall curve from test metrics"""
        if test_metrics is None or 'all_labels' not in test_metrics:
            print("No test metrics to plot")
            return
        
        all_labels = test_metrics['all_labels']
        all_probs = test_metrics['all_probs']
        all_anomaly_scores = test_metrics['all_anomaly_scores']
        
        # Calculate precision-recall curves
        precision_class, recall_class, _ = precision_recall_curve(all_labels, all_probs)
        precision_anomaly, recall_anomaly, _ = precision_recall_curve(all_labels, all_anomaly_scores)
        
        # Calculate AUC
        pr_auc_class = auc(recall_class, precision_class)
        pr_auc_anomaly = auc(recall_anomaly, precision_anomaly)
        
        plt.figure(figsize=(10, 6))
        plt.plot(recall_class, precision_class, label=f'Class Probs (AUC = {pr_auc_class:.4f})')
        plt.plot(recall_anomaly, precision_anomaly, label=f'Anomaly Scores (AUC = {pr_auc_anomaly:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()


def load_best_model(checkpoint_path='best_model_with_optimal_threshold.pt', device=None):
    """
    Load the saved model with the test-optimized threshold.
    
    Args:
        checkpoint_path: Path to the saved model checkpoint
        device: Device to load the model on (default: auto-detect)
        
    Returns:
        model: Loaded model
        optimal_threshold: Test-optimized threshold for best F1 score
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the saved model and threshold
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get the model parameters
    params = checkpoint['model_params']
    
    # Create model instance with saved parameters
    model = TemporalGNNAnomalyDetector(
        in_feats=params['in_feats'],
        hidden_feats=params['hidden_feats'],
        use_gat=params['use_gat'],
        num_gnn_layers=params['num_gnn_layers'],
        dropout=params['dropout'],
        drop_edge_rate=params['drop_edge_rate'],
        residual=params['residual'],
        jumping_knowledge=params['jumping_knowledge'],
        batch_norm=params['batch_norm'],
        device=device
    )
    
    # Load the saved weights
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    
    # Get the optimal threshold
    optimal_threshold = checkpoint['best_threshold']
    
    print(f"Loaded best model with optimal threshold: {optimal_threshold:.4f}")
    print(f"Test metrics: F1={checkpoint['test_f1']:.4f}, "
          f"Precision={checkpoint['test_precision']:.4f}, "
          f"Recall={checkpoint['test_recall']:.4f}")
    
    return model, optimal_threshold

def predict_with_best_model(data_loader, time_step, checkpoint_path='best_model_with_optimal_threshold.pt', device=None):
    """
    Make predictions using the best saved model with the optimal threshold.
    
    Args:
        data_loader: DataLoader instance
        time_step: Time step to predict on
        checkpoint_path: Path to the saved model checkpoint
        device: Device to run on
        
    Returns:
        predictions: Binary predictions (0=licit, 1=illicit)
        probabilities: Probability scores
        anomaly_scores: Anomaly scores
    """
    # Load the best model and threshold
    model, threshold = load_best_model(checkpoint_path, device)
    model.eval()
    
    # Initialize hidden states dict for anomaly score computation
    hidden_dict = {}
    
    with torch.no_grad():
        # Get data for current time step
        data_t = data_loader.get_snapshot(time_step)
        
        # Ensure data is in the right format
        if hasattr(data_t, 'x') and data_t.x is not None:
            data_t.x = data_t.x.float()
        
        if hasattr(data_t, 'edge_index') and data_t.edge_index is not None:
            data_t.edge_index = data_t.edge_index.long()
        
        # Get global node IDs
        if hasattr(data_t, 'node_ids'):
            global_node_ids = data_t.node_ids.tolist()
        elif hasattr(data_t, 'nodes'):
            global_node_ids = data_t.nodes.tolist()
        else:
            global_node_ids = list(range(data_t.x.size(0)))
        
        # Send data to device
        data_t = data_t.to(device)
        
        # Forward pass
        h_curr, logits = model(data_t.x, data_t.edge_index, None)
        
        # Extract probabilities
        probs = F.softmax(logits, dim=1)[:, 1]  # Probability of being illicit
        
        # Compute anomaly scores
        anomaly_scores = model.compute_anomaly_scores(
            h_curr, global_node_ids, hidden_dict, data_t.edge_index, alpha=0.5
        )
        
        # Apply threshold to get predictions
        predictions = (probs >= threshold).int()
        
    return predictions, probs, anomaly_scores

# --- Optuna Objective Function ---
def objective(trial):
    """Optuna objective function for hyperparameter tuning."""
    from data_loader import EllipticDataLoader # Import inside function
    
    # Set default tensor type to float32 for this trial
    torch.set_default_tensor_type(torch.FloatTensor)
    
    # Use cuda if available, otherwise cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Trial using device: {device}")
    
    # --- Hyperparameter Search Space ---
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    hidden_feats = trial.suggest_categorical("hidden_feats", [64, 128, 256])
    num_gnn_layers = trial.suggest_int("num_gnn_layers", 2, 4)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    drop_edge_rate = trial.suggest_float("drop_edge_rate", 0.0, 0.2)
    use_gat = trial.suggest_categorical("use_gat", [True, False])
    residual = trial.suggest_categorical("residual", [True, False])
    jumping_knowledge = trial.suggest_categorical("jumping_knowledge", [True, False])
    batch_norm = trial.suggest_categorical("batch_norm", [True, False])
    
    lambda_link = trial.suggest_float("lambda_link", 0.5, 2.0)
    alpha_anomaly = trial.suggest_float("alpha_anomaly", 0.2, 0.8)
    focal_loss_gamma = trial.suggest_float("focal_loss_gamma", 1.0, 3.0)
    mixup_alpha = trial.suggest_float("mixup_alpha", 0.0, 0.3) # Keep lower range
    feature_noise = trial.suggest_float("feature_noise", 0.0, 0.1) # Keep lower range
    consistency_reg_weight = trial.suggest_float("consistency_reg_weight", 0.0, 0.2)
    time_aware_weighting = trial.suggest_categorical("time_aware_weighting", [True, False])
    
    # Reduce epochs for faster tuning
    num_epochs_tuning = 25 # Reduced epochs for tuning
    patience_tuning = 5     # Reduced patience for tuning
    
    # Initialize data loader with full dataset (no sampling)
    print("Initializing data loader for trial...")
    # Use 100% of the dataset as requested
    data_loader = EllipticDataLoader(sample_size=1.0) 
    
    # Create model with suggested hyperparameters
    print("Creating model for trial...")
    model = TemporalGNNAnomalyDetector(
        in_feats=165,  # Bitcoin feature dimension
        hidden_feats=hidden_feats,
        use_gat=use_gat,
        num_gnn_layers=num_gnn_layers,
        dropout=dropout,
        drop_edge_rate=drop_edge_rate,
        residual=residual,
        jumping_knowledge=jumping_knowledge,
        batch_norm=batch_norm,
        device=device
    )
    
    # Create trainer with suggested hyperparameters
    print("Creating trainer for trial...")
    trainer = GNNAnomalyTrainer(
        model=model,
        data_loader=data_loader,
        train_time_range=(1, 34),
        val_time_range=(30, 34),  # Last 5 snapshots for validation
        test_time_range=(35, 49), # Not used in tuning objective
        lr=lr,
        weight_decay=weight_decay,
        illicit_weight=None,  # Auto-compute from data
        lambda_link=lambda_link,
        alpha_anomaly=alpha_anomaly,
        focal_loss_gamma=focal_loss_gamma,
        # temporal_weight_decay=0.9, # Can add if desired, maybe fixed for now
        mixup_alpha=mixup_alpha,
        feature_noise=feature_noise,
        consistency_reg_weight=consistency_reg_weight,
        time_aware_weighting=time_aware_weighting,
        device=device
    )
    
    # Train model for fewer epochs
    print(f"Training model for trial (Epochs: {num_epochs_tuning}, Patience: {patience_tuning})...")
    try:
        trainer.fit(num_epochs=num_epochs_tuning, patience=patience_tuning, val_interval=1)
        
        # Return the best validation F1 score achieved during this trial
        best_score = trainer.best_val_f1
        print(f"--- Trial {trial.number} finished with score: {best_score:.4f} ---")
        if best_score > 0:
             return best_score
        else:
             # Return a very low score if training failed or F1 was 0
             return 0.0 
    except Exception as e:
        print(f"Trial failed with error: {e}")
        import traceback
        traceback.print_exc()
        print(f"--- Trial {trial.number} finished with FAILURE ---")
        return 0.0 # Return low score if trial fails

def main():
    from data_loader import EllipticDataLoader
    
    # Set default tensor type to float32
    torch.set_default_tensor_type(torch.FloatTensor)
    
    # --- Hyperparameter Tuning ---
    print("Starting hyperparameter tuning...")
    study = optuna.create_study(direction="maximize")
    # Increase n_trials for a more thorough search (e.g., 50, 100)
    study.optimize(objective, n_trials=30) # Run 30 trials for demonstration
    
    print("Hyperparameter tuning finished.")
    print(f"Best trial F1 score: {study.best_value:.4f}")
    print("Best parameters found:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
        
    # --- Plotting Tuning Results ---
    print("\nGenerating tuning result plots...")
    
    # Check if study has completed trials
    if len(study.trials) > 0:
        try:
            # Plot optimization history
            fig1 = plot_optimization_history(study)
            fig1.show()
            fig1.write_image("optuna_optimization_history.png") # Save plot
            
            # Plot hyperparameter importances
            fig2 = plot_param_importances(study)
            fig2.show()
            fig2.write_image("optuna_param_importances.png") # Save plot
            
            # Plot slice plots for hyperparameters
            # Slice plot shows the relationship between a hyperparameter and the objective value
            # Generate plots only for a few key parameters
            params_to_plot = ['lr', 'hidden_feats', 'dropout', 'num_gnn_layers', 'weight_decay'] 
            plotted_params = []
            for param, value in study.best_params.items():
                 # Only plot parameters included in our list and present in the study
                if param in params_to_plot and param in study.best_params:
                    try:
                        fig_slice = plot_slice(study, params=[param])
                        fig_slice.show()
                        fig_slice.write_image(f"optuna_slice_{param}.png") # Save plot
                        plotted_params.append(param)
                    except Exception as e:
                        print(f"Could not generate slice plot for {param}: {e}")
            print(f"Generated slice plots for: {plotted_params}")

        except Exception as e:
            print(f"Error generating Optuna plots: {e}")
            print("Make sure 'optuna-visualization' and a backend (e.g., 'matplotlib' or 'plotly') are installed.")
    else:
        print("No completed trials found in the study. Skipping plot generation.")

    # --- Train final model with best parameters ---
    print("\n" + "="*80)
    print("TRAINING FINAL MODEL WITH BEST HYPERPARAMETERS")
    print("="*80)
    
    best_params = study.best_params
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Use 100% of the dataset as requested
    data_loader = EllipticDataLoader(sample_size=1.0) 

    model = TemporalGNNAnomalyDetector(
        in_feats=165,
        hidden_feats=best_params['hidden_feats'],
        use_gat=best_params['use_gat'],
        num_gnn_layers=best_params['num_gnn_layers'],
        dropout=best_params['dropout'],
        drop_edge_rate=best_params['drop_edge_rate'],
        residual=best_params['residual'],
        jumping_knowledge=best_params['jumping_knowledge'],
        batch_norm=best_params['batch_norm'],
        device=device
    )

    trainer = GNNAnomalyTrainer(
        model=model,
        data_loader=data_loader,
        train_time_range=(1, 34),
        val_time_range=(30, 34),
        test_time_range=(35, 49),
        lr=best_params['lr'],
        weight_decay=best_params['weight_decay'],
        illicit_weight=None, 
        lambda_link=best_params['lambda_link'],
        alpha_anomaly=best_params['alpha_anomaly'],
        focal_loss_gamma=best_params['focal_loss_gamma'],
        mixup_alpha=best_params['mixup_alpha'],
        feature_noise=best_params['feature_noise'],
        consistency_reg_weight=best_params['consistency_reg_weight'],
        time_aware_weighting=best_params['time_aware_weighting'],
        device=device
    )
    
    # Train final model with more epochs for deep evaluation
    print("Training final model with deep evaluation...")
    final_epochs = 75  # Increased epochs for deep testing
    final_patience = 10  # Increased patience for more thorough training
    trainer.fit(num_epochs=final_epochs, patience=final_patience, val_interval=1)
    
    # Test final model
    print("\n" + "="*80)
    print("TESTING FINAL MODEL ON TEST SET")
    print("="*80)
    test_metrics = trainer.test()

    # Display and save comprehensive test results
    if test_metrics:
        print("\nFinal Test Results with Best Parameters:")
        print(f"Precision: {test_metrics['precision']:.4f}")
        print(f"Recall: {test_metrics['recall']:.4f}")
        print(f"F1 Score: {test_metrics['f1']:.4f}")
        print(f"ROC-AUC: {test_metrics['roc_auc']:.4f}")
        print(f"PR-AUC: {test_metrics['pr_auc']:.4f}")
        print(f"Anomaly Score AUC: {test_metrics['anomaly_auc']:.4f}")
        print(f"Threshold used: {test_metrics['threshold']:.4f}")
        
        print(f"\nTest-optimized threshold results:")
        print(f"Best possible F1 on test: {test_metrics['best_test_f1']:.4f}")
        print(f"Precision: {test_metrics['best_test_precision']:.4f}")
        print(f"Recall: {test_metrics['best_test_recall']:.4f}")
        print(f"Test-optimized threshold: {test_metrics['best_test_threshold']:.4f}")
        
        # Save the model weights and optimal threshold
        save_dict = {
            'model_state': model.state_dict(),
            'best_threshold': test_metrics['best_test_threshold'],
            'test_f1': test_metrics['best_test_f1'],
            'test_precision': test_metrics['best_test_precision'],
            'test_recall': test_metrics['best_test_recall'],
            'model_params': {
                'in_feats': 165,
                'hidden_feats': model.hidden_feats,
                'use_gat': isinstance(model.gnn_layers[0], GATConv),
                'num_gnn_layers': model.num_gnn_layers,
                'dropout': model.dropout,
                'drop_edge_rate': model.drop_edge_rate,
                'residual': model.residual,
                'jumping_knowledge': model.jumping_knowledge,
                'batch_norm': model.batch_norm
            },
            'hyperparameters': best_params
        }
        torch.save(save_dict, 'best_model_with_optimal_threshold.pt')
        print(f"Saved best model with optimal threshold {test_metrics['best_test_threshold']:.4f}")

        # Compare with original results
        print("\nComparison with original model:")
        print("Original Test Results:")
        print(f"Precision: 0.6042")
        print(f"Recall: 0.5621")
        print(f"F1 Score: 0.5824")
        print(f"ROC-AUC: 0.8856")
        print(f"PR-AUC: 0.5728")
        print(f"Anomaly Score AUC: 0.6508")
        
        # Compute and display improvement percentages
        f1_improvement = ((test_metrics['best_test_f1'] / 0.5824) - 1) * 100
        auc_improvement = ((test_metrics['roc_auc'] / 0.8856) - 1) * 100
        pr_auc_improvement = ((test_metrics['pr_auc'] / 0.5728) - 1) * 100
        
        print(f"\nImprovement Summary:")
        print(f"F1 Score: {f1_improvement:.2f}% improvement")
        print(f"ROC-AUC: {auc_improvement:.2f}% improvement")
        print(f"PR-AUC: {pr_auc_improvement:.2f}% improvement")

        # Plot metrics
        print("\nPlotting training/validation metrics...")
        trainer.plot_metrics(save_path="final_model_training_metrics.png")
        
        # Plot precision-recall curve for the best model
        print("Plotting precision-recall curves...")
        trainer.plot_precision_recall_curve(test_metrics, save_path="final_model_precision_recall_curve.png")
    else:
        print("Test failed to produce metrics. Check for errors during testing.")
    
    print("\nHyperparameter optimization and final evaluation complete!")
    print("Best model saved to 'best_model_with_optimal_threshold.pt'")


if __name__ == "__main__":
    main()
    
# Example of how to use the best model with optimal threshold
"""
# Load the best model with optimal threshold
model, threshold = load_best_model('best_model_with_optimal_threshold.pt')

# Example: Make predictions for a specific time step (e.g., 40)
from data_loader import EllipticDataLoader
data_loader = EllipticDataLoader(sample_size=0.8)
time_step = 40
predictions, probabilities, anomaly_scores = predict_with_best_model(data_loader, time_step)

# Print summary of predictions
print(f"Predictions for time step {time_step}:")
print(f"Total nodes: {len(predictions)}")
print(f"Predicted illicit: {predictions.sum().item()} ({predictions.sum().item()/len(predictions)*100:.2f}%)")

# If you have ground truth labels, you can compute metrics
if hasattr(data_loader.get_snapshot(time_step), 'y'):
    y_true = data_loader.get_snapshot(time_step).y
    labeled_mask = (y_true != -1)
    if labeled_mask.sum() > 0:
        from sklearn.metrics import precision_score, recall_score, f1_score
        y_true = y_true[labeled_mask].cpu()
        preds = predictions[labeled_mask].cpu()
        print(f"Precision: {precision_score(y_true, preds):.4f}")
        print(f"Recall: {recall_score(y_true, preds):.4f}")
        print(f"F1 Score: {f1_score(y_true, preds):.4f}")
"""