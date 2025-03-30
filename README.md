# Anomaly Detection in Large-Scale Evolving Networks using Temporal GNNs

This project implements anomaly detection methods for the Elliptic Bitcoin transaction dataset using Temporal Graph Neural Networks (GNNs). The implementation includes two complementary approaches:

1. A prediction-based encoder-decoder that forecasts future graph states and flags anomalies via high reconstruction error
2. An online scoring system that evaluates new edges or node behaviors against historical embedding patterns

## Dataset

The Elliptic dataset is a graph of Bitcoin transactions over time, containing:
- 203,769 transaction nodes
- 234,355 directed payment edges
- 49 distinct time steps
- Node features (165 attributes)
- Labels: licit (legal) and illicit (malicious) transactions

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the Elliptic dataset and place the following files in the `elliptic_data` directory:
- `elliptic_txs_features.csv`
- `elliptic_txs_edgelist.csv`
- `elliptic_txs_classes.csv`

## Project Structure

- `data_loader.py`: Handles loading and preprocessing the Elliptic dataset
- `models.py`: Contains the GNN and temporal model implementations
- `train.py`: Training and evaluation scripts
- `requirements.txt`: Project dependencies

## Usage

To train and evaluate the models:

```bash
python train.py
```

This will:
1. Load and preprocess the dataset
2. Train both the temporal GNN and online detector
3. Evaluate the models on the test set
4. Visualize detected anomalies

## Model Details

### Temporal GNN
- Uses a GCN encoder to process each graph snapshot
- GRU-based temporal modeling to capture evolution
- Decoder predicts future edge existence
- Anomalies detected via reconstruction error

### Online Detector
- Maintains node embeddings that evolve over time
- Scores new edges based on embedding similarity
- Lightweight and suitable for real-time detection
- No retraining needed for new data

## Evaluation Metrics

The models are evaluated using:
- Precision
- Recall
- F1-Score
- ROC-AUC

## Visualization

The code includes visualization tools to:
- Plot graph structure with anomalous nodes highlighted
- Project node embeddings to 2D space using t-SNE
- Show temporal patterns in anomaly detection

## References

1. Elliptic Dataset: [Bitcoin Transaction Network](https://www.kaggle.com/ellipticco/elliptic-data-set)
2. Temporal GNNs: [EvolveGCN](https://arxiv.org/abs/1902.10191)
3. Anomaly Detection: [AddGraph](https://arxiv.org/abs/1904.12072)

## License

MIT License 