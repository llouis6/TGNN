# Anomaly Detection in Large-Scale Evolving Networks using Temporal GNNs

This project implements a comprehensive framework for anomaly detection in the Elliptic Bitcoin transaction dataset using advanced Temporal Graph Neural Networks (GNNs). The implementation combines hyperparameter optimization with a dual-pipeline approach for robust anomaly detection.

## Overview

Our approach addresses three key challenges in temporal network anomaly detection:
- **Real-time Processing**: Efficient algorithms for streaming data without storing full history
- **Structural Evolution**: Capturing unusual patterns in how network structure changes over time
- **Imperfect Datasets**: Handling heavily imbalanced data with scarce labeling

## Dataset

The Elliptic dataset is a graph of Bitcoin transactions over time, containing:
- **203,769** transaction nodes
- **234,355** directed payment edges
- **49** distinct time steps
- **165** node features (transaction attributes)
- **Labels**: licit (legal) and illicit (malicious) transactions
- **Class Distribution**: ~2% illicit, ~23% licit, ~75% unknown

## Key Features

### Dual-Pipeline Architecture
1. **Prediction-based Encoder-Decoder**: Forecasts future graph states and flags anomalies via reconstruction error
2. **Online Embedding Similarity Detector**: Evaluates new edges/node behaviors against historical embedding patterns

### Advanced Techniques
- **Hyperparameter Optimization**: Automated tuning using Optuna (30 trials)
- **Advanced Regularization**: Mixup, feature noise, DropEdge, focal loss, consistency regularization
- **Temporal Modeling**: GRU-based sequence modeling with attention mechanisms
- **Semi-supervised Learning**: Combines labeled and unlabeled data effectively

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. **Download the Elliptic dataset** from [Kaggle](https://www.kaggle.com/ellipticco/elliptic-data-set) and place the following files in the `elliptic_data` directory:
- `elliptic_txs_features.csv` (657 MB - requires Kaggle account)
- `elliptic_txs_edgelist.csv`
- `elliptic_txs_classes.csv`

**Note**: The dataset files are not included in this repository due to size limitations. You must download them separately from Kaggle.

## Project Structure

```
temporal-gnn-anomaly-detection/
├── README.md                           # This file
├── requirements.txt                    # Project dependencies
├── data_loader.py                      # Dataset loading and preprocessing
├── models.py                           # Baseline model implementations
├── temporal_gnn_main.py               # Main implementation with hyperparameter tuning
├── baseline_training.py               # Baseline training script
├── inference.py                       # Model inference and prediction
├── elliptic_data/                     # Dataset directory
│   ├── elliptic_txs_features.csv
│   ├── elliptic_txs_edgelist.csv
│   └── elliptic_txs_classes.csv
├── tuning_results/                    # Hyperparameter optimization results
│   └── best_params_*.json
├── results/                           # Trained models and visualizations
│   ├── best_model_with_optimal_threshold.pt
│   ├── final_model_training_metrics.png
│   ├── final_model_precision_recall_curve.png
│   └── temporal_*.png
└── __pycache__/                       # Python cache (ignored by git)
```

## Usage

### Main Implementation (Recommended)
To run the complete hyperparameter optimization and training pipeline:

```bash
python temporal_gnn_main.py
```

This will:
1. Run 30 trials of hyperparameter optimization using Optuna
2. Generate optimization history and parameter importance plots
3. Train the final model with best hyperparameters (75 epochs)
4. Evaluate on test set with comprehensive metrics
5. Save the best model and generate visualization plots

### Baseline Training
To run the baseline temporal GNN implementation:

```bash
python baseline_training.py
```

### Inference
To make predictions using a trained model:

```bash
python inference.py
```

## Model Architecture

### Temporal GNN Anomaly Detector
- **Encoder**: GraphSAGE or GAT layers with optional residual connections
- **Temporal Module**: GRU-based sequence modeling for temporal evolution
- **Decoder**: Link prediction and node classification heads
- **Anomaly Scoring**: Combines reconstruction error and embedding similarity

### Key Components
- **Jumping Knowledge**: Aggregates representations from multiple GNN layers
- **Batch Normalization**: Stabilizes training dynamics
- **DropEdge**: Random edge dropping for regularization
- **Focal Loss**: Handles class imbalance effectively
- **Mixup Augmentation**: Improves generalization

## Results

### Performance Comparison

| Method | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|--------|-----------|--------|----------|---------|---------|
| Node2Vec + LOF (Static) | 0.57 | 0.14 | 0.25 | 0.69 | - |
| Logistic Regression | 0.77 | 0.94 | 0.67 | 0.86 | 0.24 |
| **Temporal GNN (Ours)** | **0.70** | **0.67** | **0.68** | **0.90** | **0.52** |

### Key Achievements
- **Best ROC-AUC**: 0.90 (superior ranking of illicit vs licit transactions)
- **Balanced Performance**: F1-score of 0.68 with balanced precision/recall
- **Temporal Advantage**: Captures evolving network patterns that static methods miss
- **Robust Regularization**: Advanced techniques prevent overfitting

### Training Characteristics
- **Validation F1**: Improves from 0.55 to 0.83 (epoch 22)
- **Early Stopping**: Triggered at epoch 29 to prevent overfitting
- **Distribution Drift**: Performance drops on later time steps (35-49) due to evolving transaction patterns

## Evaluation Metrics

The models are evaluated using:
- **Precision**: Fraction of flagged anomalies that are truly illicit
- **Recall**: Fraction of true illicit nodes that were detected
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve (threshold-independent)
- **PR-AUC**: Area under Precision-Recall curve (sensitive to class imbalance)

## Hyperparameter Optimization

The main implementation uses Optuna for automated hyperparameter tuning:

### Search Space
- **Learning Rate**: 1e-4 to 1e-2 (log scale)
- **Hidden Features**: [64, 128, 256]
- **GNN Layers**: 2-4 layers
- **Dropout**: 0.1-0.5
- **DropEdge Rate**: 0.0-0.2
- **Regularization**: Mixup alpha, feature noise, consistency weight
- **Architecture**: GAT vs GraphSAGE, residual connections, batch norm

### Optimization Process
- **30 Trials**: Comprehensive search of hyperparameter space
- **25 Epochs per Trial**: Fast evaluation for tuning
- **Early Stopping**: Prevents overfitting during optimization
- **Validation F1**: Primary optimization objective

## Visualization

The framework includes comprehensive visualization tools:
- **Optimization History**: Shows improvement over trials
- **Parameter Importance**: Identifies most influential hyperparameters
- **Training Metrics**: Loss curves and validation performance
- **Precision-Recall Curves**: Model performance across thresholds
- **Slice Plots**: Hyperparameter sensitivity analysis

## Challenges and Limitations

### Dataset Challenges
- **Class Imbalance**: Only 2% illicit nodes make detection difficult
- **Unknown Labels**: 75% of nodes have unknown status
- **Temporal Drift**: Transaction patterns evolve over time
- **Scale**: Large dataset requires careful sampling strategies

### Model Limitations
- **Overfitting**: Performance drops from validation to test
- **Distribution Shift**: Future time steps differ from training data
- **Threshold Sensitivity**: Small changes significantly impact precision/recall

## Future Work

### Technical Improvements
- **Online Learning**: Incremental model updates for streaming data
- **Transformer Architecture**: More flexible temporal modeling
- **Contrastive Learning**: Better utilization of unlabeled data
- **Adaptive Sampling**: Focus on suspicious subgraphs

### Evaluation Extensions
- **Multi-dataset Validation**: Test on other dynamic graph datasets
- **Continuous-time Models**: Handle asynchronous events
- **Explanation Modules**: Interpretability for detected anomalies
- **Real-time Deployment**: Production-ready inference pipeline

## References

1. Elliptic Dataset: [Bitcoin Transaction Network](https://www.kaggle.com/ellipticco/elliptic-data-set)
2. Temporal GNNs: [EvolveGCN](https://arxiv.org/abs/1902.10191)
3. Anomaly Detection: [AddGraph](https://arxiv.org/abs/1904.12072)
4. Optuna: [Hyperparameter Optimization Framework](https://optuna.org/)

## Authors

- **Luke Cashion** - McGill University (luke.cashion@mail.mcgill.ca)
- **Luca Louis** - McGill University (luca.louis@mail.mcgill.ca)

## License

MIT License

---

*This project was developed as part of research in temporal graph anomaly detection for financial transaction networks.*