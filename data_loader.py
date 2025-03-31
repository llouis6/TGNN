import os
import pandas as pd
import torch
from torch_geometric.data import Data
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

class EllipticDataLoader:
    def __init__(self, data_dir: str = "./elliptic_data", sample_size: Optional[float] = None):
        """Initialize the data loader for the Elliptic Bitcoin dataset.
        
        Args:
            data_dir: Directory containing the dataset files
            sample_size: Fraction of data to sample (0.0 to 1.0). If None, use all data.
        """
        self.data_dir = data_dir
        self.sample_size = sample_size
        self.feat_path = os.path.join(data_dir, "elliptic_txs_features.csv")
        self.edge_path = os.path.join(data_dir, "elliptic_txs_edgelist.csv")
        self.class_path = os.path.join(data_dir, "elliptic_txs_classes.csv")
        
        # Load data
        self._load_data()
        
    def _load_data(self):
        """Load and preprocess the dataset files."""
        # Load CSV files
        self.feat_df = pd.read_csv(self.feat_path, header=None)
        self.edge_df = pd.read_csv(self.edge_path)
        self.class_df = pd.read_csv(self.class_path)
        
        # Rename first two columns of features
        self.feat_df = self.feat_df.rename(columns={0: "txId", 1: "time_step"})
        
        # Map class labels: '2'->0 (licit), '1'->1 (illicit), 'unknown'-> -1
        class_map = {"2": 0, "1": 1, "unknown": -1}
        self.class_df['class'] = self.class_df['class'].map(class_map)
        
        # Sample the data if requested
        if self.sample_size is not None and self.sample_size < 1.0:
            # Sample transactions
            sampled_txs = self.feat_df['txId'].sample(frac=self.sample_size, random_state=42)
            
            # Filter features
            self.feat_df = self.feat_df[self.feat_df['txId'].isin(sampled_txs)]
            
            # Filter edges where both nodes are in the sample
            self.edge_df = self.edge_df[
                self.edge_df['txId1'].isin(sampled_txs) & 
                self.edge_df['txId2'].isin(sampled_txs)
            ]
            
            # Filter classes
            self.class_df = self.class_df[self.class_df['txId'].isin(sampled_txs)]
            
            print(f"Sampled {len(sampled_txs)} transactions")
            print(f"Sampled {len(self.edge_df)} edges")
        
        # Merge class labels into features DataFrame
        self.feat_df = self.feat_df.merge(self.class_df, how="left", 
                                        left_on="txId", right_on="txId")
        self.feat_df['class'] = self.feat_df['class'].fillna(-1)
        
        # Create node index mapping
        self.txid_to_index = {txId: idx for idx, txId in enumerate(self.feat_df['txId'])}
        self.index_to_txid = {idx: txId for txId, idx in self.txid_to_index.items()}
        
        # Convert edge list to index-based
        self.edge_df['src_idx'] = self.edge_df['txId1'].map(self.txid_to_index)
        self.edge_df['dst_idx'] = self.edge_df['txId2'].map(self.txid_to_index)
        self.edges_indexed = self.edge_df[['src_idx', 'dst_idx']].dropna().astype(int).values.T
        
        # Group nodes and edges by time step
        self._group_by_time()
        
    def _group_by_time(self):
        """Group nodes and edges by time step."""
        # Group nodes by time
        self.nodes_by_time = defaultdict(list)
        for idx, t in enumerate(self.feat_df['time_step']):
            self.nodes_by_time[int(t)].append(idx)
            
        # Group edges by time (assign to destination node's time)
        self.edges_by_time = defaultdict(list)
        for src, dst in zip(self.edges_indexed[0], self.edges_indexed[1]):
            t_dst = int(self.feat_df.iloc[dst]['time_step'])
            self.edges_by_time[t_dst].append((src, dst))
            
        # Create graph snapshots
        self.graph_snapshots = {}
        for t in range(1, 50):  # time steps 1..49
            node_idx_list = self.nodes_by_time[t]
            if not node_idx_list:  # Skip if no nodes in this time step
                continue
            
            # Create local node index mapping for this snapshot
            global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(node_idx_list)}
            local_to_global = {local_idx: global_idx for global_idx, local_idx in global_to_local.items()}
            
            # Convert global edge indices to local indices
            edge_index = None
            if t in self.edges_by_time:
                edge_list = self.edges_by_time[t]
                local_edges = []
                for src, dst in edge_list:
                    if src in global_to_local and dst in global_to_local:
                        local_edges.append([global_to_local[src], global_to_local[dst]])
                if local_edges:  # Only create edge_index if there are valid edges
                    edge_index = list(zip(*local_edges))
            
            # Extract node features and labels for this snapshot
            node_features = self.feat_df.iloc[node_idx_list, 2:-1].to_numpy()
            labels = self.feat_df.iloc[node_idx_list]['class'].to_numpy()
            
            self.graph_snapshots[t] = {
                "nodes": node_idx_list,
                "edge_index": edge_index,
                "x": node_features,
                "y": labels,
                "global_to_local": global_to_local,
                "local_to_global": local_to_global
            }
            
    def get_snapshot(self, t: int) -> Data:
        """Get a PyTorch Geometric Data object for time step t.
        
        Args:
            t: Time step (1-49)
            
        Returns:
            PyG Data object containing the graph at time t
        """
        if t not in self.graph_snapshots:
            raise ValueError(f"Time step {t} not found in dataset")
            
        snapshot = self.graph_snapshots[t]
        x = torch.tensor(snapshot['x'], dtype=torch.float)
        y = torch.tensor(snapshot['y'], dtype=torch.long)
        
        # Create edge_index tensor if edges exist
        if snapshot['edge_index'] is not None:
            edge_index = torch.tensor(snapshot['edge_index'], dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # Create Data object with all necessary attributes
        data = Data(x=x, edge_index=edge_index, y=y)
        data.nodes = torch.tensor(snapshot['nodes'], dtype=torch.long)
        data.global_to_local = snapshot['global_to_local']
        data.local_to_global = snapshot['local_to_global']
        
        return data
    
    def get_train_test_split(self, train_end: int = 34) -> Tuple[List[Data], List[Data]]:
        """Get training and test splits of the dataset.
        
        Args:
            train_end: Last time step to include in training (default: 34)
            
        Returns:
            Tuple of (train_data_list, test_data_list)
        """
        train_data = []
        test_data = []
        
        for t in range(1, 50):
            data = self.get_snapshot(t)
            if t <= train_end:
                train_data.append(data)
            else:
                test_data.append(data)
                
        return train_data, test_data
    
    def get_node_feature_dim(self) -> int:
        """Get the dimension of node features."""
        return self.feat_df.shape[1] - 3  # excluding txId, time_step, class
    
    def get_num_nodes(self) -> int:
        """Get total number of nodes in the dataset."""
        return len(self.feat_df) 