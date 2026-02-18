# Network Flow Data Preprocessing

Two preprocessing scripts are available for network flow data analysis:

1. **`preprocess_network_data_optimized.py`** ⭐ **RECOMMENDED** - Optimized for UNSW-NB15 and similar datasets
2. **`preprocess_network_data.py`** - Generic script for various CSV formats

## Quick Start

### 1. Setup Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On Linux/Mac
# OR
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Preprocessing

```bash
# Test on 10,000 rows first
./venv/bin/python preprocess_network_data_optimized.py reduced_data_1.csv --nrows 10000 -o test_output.csv

# Process full dataset
./venv/bin/python preprocess_network_data_optimized.py reduced_data_1.csv -o preprocessed_data_1.csv
```

## Features

### Optimized Script (`preprocess_network_data_optimized.py`)

✅ **Automatic Feature Extraction** - Extracts 19+ features from UNSW-NB15 format  
✅ **Derived Features** - Creates Bytes_Per_Packet, Packets_Per_Second, Src_Dst ratios  
✅ **Protocol & State Encoding** - One-hot encoding for categorical features  
✅ **IP Address Encoding** - Label encoding for source/destination IPs  
✅ **Port Handling** - Optional encoding or removal of port numbers  
✅ **Normalization** - Optional StandardScaler normalization  
✅ **Label Preservation** - Keeps attack labels for supervised learning  

### Extracted Features

**Basic Features:**
- Source_IP, Destination_IP (label encoded)
- Protocol (one-hot encoded: tcp, udp, arp, icmp)
- State (one-hot encoded: CON, REQ, RST, etc.)
- Packet_Count, Duration, Bytes_Transferred
- Source_Packets, Destination_Packets
- Source_Bytes, Destination_Bytes
- Rate, Source_Rate, Destination_Rate

**Derived Features:**
- Bytes_Per_Packet
- Packets_Per_Second
- Src_Dst_Packet_Ratio
- Src_Dst_Byte_Ratio

**Labels (preserved for ML):**
- Label (0=normal, 1=attack)
- Attack_Category (e.g., DoS, Reconnaissance)
- Attack_Subcategory (e.g., HTTP, DNS)

## Usage Examples

### Basic Usage

```bash
# Process with default settings
./venv/bin/python preprocess_network_data_optimized.py reduced_data_1.csv
```

### Advanced Options

```bash
# Test on subset of data
./venv/bin/python preprocess_network_data_optimized.py reduced_data_1.csv --nrows 50000

# Specify output file
./venv/bin/python preprocess_network_data_optimized.py reduced_data_1.csv -o my_output.csv

# Keep IP addresses as strings (don't encode)
./venv/bin/python preprocess_network_data_optimized.py reduced_data_1.csv --no-encode-ips

# Include port encoding
./venv/bin/python preprocess_network_data_optimized.py reduced_data_1.csv --encode-ports

# Normalize features (StandardScaler)
./venv/bin/python preprocess_network_data_optimized.py reduced_data_1.csv --normalize

# Skip derived features
./venv/bin/python preprocess_network_data_optimized.py reduced_data_1.csv --no-derived
```

### Process All Files

```bash
# Process all 4 CSV files
for i in 1 2 3 4; do
    ./venv/bin/python preprocess_network_data_optimized.py \
        reduced_data_$i.csv \
        -o preprocessed_data_$i.csv \
        --normalize
done
```

## Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `input_file` | Path to input CSV file | Required |
| `--output`, `-o` | Output file path | `preprocessed_features.csv` |
| `--nrows` | Number of rows to load (for testing) | All rows |
| `--no-encode-ips` | Skip IP address encoding | Encode IPs |
| `--encode-ports` | Encode port numbers | Drop ports |
| `--normalize` | Normalize numerical features | No normalization |
| `--no-derived` | Skip derived feature creation | Create derived features |

## Output Format

The preprocessed CSV will have approximately **24 columns**:

```
Source_IP, Destination_IP, Packet_Count, Duration, Bytes_Transferred,
Source_Packets, Destination_Packets, Source_Bytes, Destination_Bytes,
Rate, Source_Rate, Destination_Rate, Bytes_Per_Packet, Packets_Per_Second,
Src_Dst_Packet_Ratio, Src_Dst_Byte_Ratio, Protocol_arp, Protocol_tcp,
State_CON, State_REQ, State_RST, Label, Attack_Category, Attack_Subcategory
```

## Example: Complete ML Pipeline

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Preprocess data with normalization
./venv/bin/python preprocess_network_data_optimized.py \
    reduced_data_1.csv \
    -o preprocessed_data_1.csv \
    --normalize

# 3. Use in Python for ML
python
```

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load preprocessed data
df = pd.read_csv('preprocessed_data_1.csv')

# Separate features and labels
X = df.drop(columns=['Label', 'Attack_Category', 'Attack_Subcategory'])
y = df['Label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")
```

## Dataset Compatibility

The optimized script is designed for datasets with these columns:

| Column | Description | Required |
|--------|-------------|----------|
| `saddr` | Source IP address | ✅ |
| `daddr` | Destination IP address | ✅ |
| `proto` | Protocol (tcp, udp, etc.) | ✅ |
| `pkts` | Total packets | ✅ |
| `bytes` | Total bytes | ✅ |
| `dur` | Flow duration | ✅ |
| `spkts`, `dpkts` | Source/dest packets | Optional |
| `sbytes`, `dbytes` | Source/dest bytes | Optional |
| `rate`, `srate`, `drate` | Transfer rates | Optional |
| `state` | Connection state | Optional |
| `sport`, `dport` | Source/dest ports | Optional |
| `attack` | Attack label (0/1) | Optional |
| `category` | Attack category | Optional |

## Performance Notes

- Processing 1M rows takes approximately 2-5 minutes
- Memory usage: ~500MB-1GB depending on dataset size
- Use `--test` flag to process only 10,000 rows for quick testing

---

## Graph Neural Network (GNN) for Attack Detection

### Overview

The GNN model converts network flow data into graphs and uses Graph Convolutional Networks (GCN) for intrusion detection.

**Key Concept:**
- **Nodes**: IP addresses (source and destination)
- **Edges**: Network flows between IPs
- **Node Features**: Degree, traffic volume, packet statistics
- **Task**: Classify nodes as Normal or Attack

### Quick Start

```bash
# 1. Install GNN dependencies
source venv/bin/activate
pip install torch torch-geometric networkx matplotlib seaborn

# 2. Test graph builder
python graph_builder.py test_preprocessed.csv

# 3. Train GNN model (test mode - 10 epochs)
python train_gnn.py --data test_preprocessed.csv --test-mode

# 4. Full training (200 epochs with early stopping)
python train_gnn.py --data preprocessed_data_1.csv --epochs 200 --save-model gnn_model.pth
```

### GNN Architecture

```
Input (5 features per node)
  ↓
GCN Layer 1 (5 → 64)
  ↓
ReLU + Dropout
  ↓
GCN Layer 2 (64 → 32)
  ↓
ReLU + Dropout
  ↓
Dense Layer (32 → 2)
  ↓
Output (Normal/Attack)
```

### Node Features

1. **Degree**: Number of connections
2. **Total Packets**: Sum of packets sent/received
3. **Total Bytes**: Sum of bytes transferred
4. **Average Packet Size**: Bytes per packet
5. **Malicious Ratio**: Proportion of malicious connections

### Command-Line Options

```bash
python train_gnn.py \
  --data preprocessed_data_1.csv \    # Input data
  --epochs 200 \                       # Training epochs
  --lr 0.001 \                         # Learning rate
  --hidden-dim 64 \                    # Hidden layer size
  --dropout 0.5 \                      # Dropout rate
  --gnn-type GCN \                     # GCN, GAT, or SAGE
  --save-model model.pth \             # Save trained model
  --test-mode                          # Quick test (10 epochs)
```

### Expected Performance

- **Target Accuracy**: >90%
- **Training Time**: 2-5 minutes for 10K samples
- **Graph Size**: ~7-10 unique IPs per 10K flows
- **Edges**: ~8K-9K connections

### Model Variants

**1. GCN (Graph Convolutional Network)** - Default, stable and fast
```bash
python train_gnn.py --gnn-type GCN
```

**2. GAT (Graph Attention Network)** - Uses attention mechanism
```bash
python train_gnn.py --gnn-type GAT --hidden-dim 16
```

**3. GraphSAGE** - Scalable for large graphs
```bash
python train_gnn.py --gnn-type SAGE
```

### Output Files

- `training_history_YYYYMMDD_HHMMSS.png` - Loss and accuracy plots
- `confusion_matrix_YYYYMMDD_HHMMSS.png` - Confusion matrix visualization
- `gnn_model.pth` - Saved model checkpoint (if --save-model specified)

### Tips for Best Results

1. **Time-window aggregation**: Group flows into 10-60 second windows
2. **Feature normalization**: Enabled by default in graph builder
3. **Class balancing**: Use `gnn_utils.balance_dataset()` for imbalanced data
4. **Hyperparameter tuning**: Adjust `--hidden-dim` and `--dropout`
5. **Early stopping**: Automatically stops when validation accuracy plateaus

### Example: Complete ML Pipeline

```python
from graph_builder import build_graph_from_csv
from gnn_model import create_model
from train_gnn import GNNTrainer, create_train_val_test_masks

# 1. Build graph
data, builder = build_graph_from_csv('preprocessed_data_1.csv')

# 2. Create splits
train_mask, val_mask, test_mask = create_train_val_test_masks(data.num_nodes)

# 3. Create and train model
model = create_model(input_dim=data.num_features, gnn_type='GCN')
trainer = GNNTrainer(model, learning_rate=0.001)
history = trainer.train(data, train_mask, val_mask, epochs=200)

# 4. Evaluate
predictions, true_labels = trainer.get_predictions(data, test_mask)
```

---

## Troubleshooting

### Memory Issues
If processing large files causes memory issues:
```bash
# Process in chunks using --nrows
./venv/bin/python preprocess_network_data_optimized.py reduced_data_1.csv --nrows 100000
```

### Mixed Type Warnings
The warning about mixed types in port columns is normal and handled automatically.

### Virtual Environment Not Found
```bash
# Recreate virtual environment
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

## Performance Tips

1. **Test first**: Use `--nrows 10000` to test on a subset
2. **Monitor memory**: Large files (>1M rows) may need chunking
3. **Skip normalization**: Only normalize if your ML algorithm requires it
4. **Drop ports**: Unless needed, ports increase dimensionality

## Next Steps

After preprocessing, your data is ready for:
- ✅ Machine Learning (scikit-learn, XGBoost, etc.)
- ✅ Deep Learning (TensorFlow, PyTorch)
- ✅ Anomaly Detection
- ✅ Network Intrusion Detection Systems (NIDS)
- ✅ Statistical Analysis
