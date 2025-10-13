import os
import pandas as pd
import torch
from torch_geometric.data import Data
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np

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
        print("Starting data loading process...")
        self._load_data()
        
    def _load_data(self):
        """Load and preprocess the dataset files."""
        # Load CSV files
        print("Loading CSV files...")
        self.feat_df = pd.read_csv(self.feat_path, header=None)
        print(f"Loaded features: {self.feat_df.shape[0]} transactions, {self.feat_df.shape[1]} features")
        
        self.edge_df = pd.read_csv(self.edge_path)
        print(f"Loaded edges: {self.edge_df.shape[0]} edges")
        
        self.class_df = pd.read_csv(self.class_path)
        print(f"Loaded classes: {self.class_df.shape[0]} transaction labels")
        
        # Rename first two columns of features
        self.feat_df = self.feat_df.rename(columns={0: "txId", 1: "time_step"})
        
        # Map class labels: '2'->0 (licit), '1'->1 (illicit), 'unknown'-> -1
        class_map = {"2": 0, "1": 1, "unknown": -1}
        self.class_df['class'] = self.class_df['class'].map(class_map)
        
        # Sample the data if requested
        if self.sample_size is not None and self.sample_size < 1.0:
            print(f"Sampling {self.sample_size:.2%} of the data...")
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
        print("Merging class labels with features...")
        self.feat_df = self.feat_df.merge(self.class_df, how="left", 
                                        left_on="txId", right_on="txId")
        self.feat_df['class'] = self.feat_df['class'].fillna(-1)
        
        # Class distribution
        class_counts = self.feat_df['class'].value_counts()
        print("Class distribution:")
        print(f"  Licit (0): {class_counts.get(0, 0)}")
        print(f"  Illicit (1): {class_counts.get(1, 0)}")
        print(f"  Unknown (-1): {class_counts.get(-1, 0)}")
        
        # Create node index mapping
        print("Creating node index mapping...")
        self.txid_to_index = {txId: idx for idx, txId in enumerate(self.feat_df['txId'])}
        self.index_to_txid = {idx: txId for txId, idx in self.txid_to_index.items()}
        
        # Convert edge list to index-based
        print("Converting edge list to index-based format...")
        self.edge_df['src_idx'] = self.edge_df['txId1'].map(self.txid_to_index)
        self.edge_df['dst_idx'] = self.edge_df['txId2'].map(self.txid_to_index)
        self.edges_indexed = self.edge_df[['src_idx', 'dst_idx']].dropna().astype(int).values.T
        
        # Group nodes and edges by time step
        print("Grouping nodes and edges by time step...")
        self._group_by_time()
        print("Data loading and preprocessing complete!")
        
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
        print("Creating graph snapshots for each time step:")
        for t in range(1, 50):  # time steps 1..49
            node_idx_list = self.nodes_by_time[t]
            if not node_idx_list:  # Skip if no nodes in this time step
                continue
            
            # Create local node index mapping for this snapshot
            global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(node_idx_list)}
            local_to_global = {local_idx: global_idx for global_idx, local_idx in global_to_local.items()}
            
            # Convert global edge indices to local indices
            edge_index = None
            num_edges = 0
            if t in self.edges_by_time:
                edge_list = self.edges_by_time[t]
                local_edges = []
                for src, dst in edge_list:
                    if src in global_to_local and dst in global_to_local:
                        local_edges.append([global_to_local[src], global_to_local[dst]])
                if local_edges:  # Only create edge_index if there are valid edges
                    edge_index = list(zip(*local_edges))
                    num_edges = len(local_edges)
            
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
            
            # Count labels for this time step
            label_counts = pd.Series(labels).value_counts().to_dict()
            illicit = label_counts.get(1, 0)
            licit = label_counts.get(0, 0)
            unknown = label_counts.get(-1, 0)
            
            print(f"  Time step {t}: {len(node_idx_list)} nodes, {num_edges} edges | "
                  f"Licit: {licit}, Illicit: {illicit}, Unknown: {unknown}")
            
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
            if t not in self.graph_snapshots:
                continue
                
            data = self.get_snapshot(t)
            if t <= train_end:
                train_data.append(data)
            else:
                test_data.append(data)
        
        print(f"Train-test split: {len(train_data)} time steps for training, {len(test_data)} for testing")
        return train_data, test_data
    
    def get_node_feature_dim(self) -> int:
        """Get the dimension of node features."""
        return self.feat_df.shape[1] - 3  # excluding txId, time_step, class
    
    def get_num_nodes(self) -> int:
        """Get total number of nodes in the dataset."""
        return len(self.feat_df) 
    
    def validate_data(self):
        """Validate the loaded data and print information to show proper handling."""
        print("\n===== Data Validation =====")
        
        # Check for missing values
        missing_features = self.feat_df.iloc[:, 2:-1].isna().sum().sum()
        missing_labels = self.feat_df['class'].isna().sum()
        print(f"Missing feature values: {missing_features}")
        print(f"Missing label values: {missing_labels}")
        
        # Verify edge consistency
        valid_edges = 0
        invalid_edges = 0
        for src, dst in zip(self.edges_indexed[0], self.edges_indexed[1]):
            if src in self.index_to_txid and dst in self.index_to_txid:
                valid_edges += 1
            else:
                invalid_edges += 1
        print(f"Valid edges: {valid_edges}, Invalid edges: {invalid_edges}")
        
        # Verify time step consistency
        time_steps = sorted(self.nodes_by_time.keys())
        print(f"Time steps present: {time_steps}")
        print(f"Number of time steps: {len(time_steps)}")
        
        # Sample data examples
        print("\nExample transactions (first 5):")
        sample_txs = self.feat_df.head(5)[['txId', 'time_step', 'class']]
        print(sample_txs)
        
        # Feature statistics
        print("\nFeature statistics (first 5 features):")
        feature_stats = self.feat_df.iloc[:, 2:7].describe().round(3)
        print(feature_stats)
        
        print("\n===== Data validation complete =====")
        return True
    
    def visualize_data_statistics(self, save_path=None):
        """Visualize key statistics about the dataset."""
        print("\n===== Visualizing Data Statistics =====")
        
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Class distribution
        class_counts = self.feat_df['class'].value_counts().reindex([0, 1, -1], fill_value=0)
        class_names = ['Licit', 'Illicit', 'Unknown']
        axs[0, 0].bar(class_names, class_counts.values, color=['green', 'red', 'gray'])
        axs[0, 0].set_title('Class Distribution')
        axs[0, 0].set_ylabel('Count')
        for i, v in enumerate(class_counts.values):
            axs[0, 0].text(i, v + 10, str(v), ha='center')
        
        # 2. Transactions per time step
        time_step_counts = self.feat_df['time_step'].value_counts().sort_index()
        axs[0, 1].plot(time_step_counts.index, time_step_counts.values, marker='o')
        axs[0, 1].set_title('Transactions per Time Step')
        axs[0, 1].set_xlabel('Time Step')
        axs[0, 1].set_ylabel('Number of Transactions')
        
        # 3. Edges per time step
        edge_counts = {t: len(edges) for t, edges in self.edges_by_time.items()}
        time_steps = sorted(edge_counts.keys())
        edge_values = [edge_counts.get(t, 0) for t in time_steps]
        axs[1, 0].plot(time_steps, edge_values, marker='x', color='orange')
        axs[1, 0].set_title('Edges per Time Step')
        axs[1, 0].set_xlabel('Time Step')
        axs[1, 0].set_ylabel('Number of Edges')
        
        # 4. Class distribution over time
        time_steps = sorted(self.nodes_by_time.keys())
        illicit_by_time = []
        licit_by_time = []
        
        for t in time_steps:
            nodes = self.nodes_by_time[t]
            labels = self.feat_df.iloc[nodes]['class'].values
            illicit_count = np.sum(labels == 1)
            licit_count = np.sum(labels == 0)
            illicit_by_time.append(illicit_count)
            licit_by_time.append(licit_count)
        
        axs[1, 1].plot(time_steps, licit_by_time, marker='o', label='Licit', color='green')
        axs[1, 1].plot(time_steps, illicit_by_time, marker='o', label='Illicit', color='red')
        axs[1, 1].set_title('Class Distribution over Time')
        axs[1, 1].set_xlabel('Time Step')
        axs[1, 1].set_ylabel('Count')
        axs[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        print("===== Visualization complete =====")

    #run file with "python -c "from data_loader import EllipticDataLoader; loader = EllipticDataLoader(sample_size=0.1); loader.validate_data(); loader.visualize_data_statistics(); print('Done!')""

if __name__ == "__main__":
    print("Running EllipticDataLoader with sample_size=0.1")
    loader = EllipticDataLoader(sample_size=0.1)
    loader.validate_data()
    loader.visualize_data_statistics()
    print("Done!")