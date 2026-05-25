# GATs-HAR

This repository contains code for multi-frequency volatility forecasting with GATs-HAR, together with ablation experiments, Diebold-Mariano tests, and economic value evaluation based on VaR and hedging effectiveness.

## Project Structure

```text
.
├─ ablation/
│  ├─ run_ablation_high.py
│  └─ run_ablation_low.py
├─ config/
│  ├─ GNN_param.yaml
│  ├─ GNN_param_optuna.yaml
│  ├─ ablation_full.yaml
│  ├─ high_freq_he.yaml
│  ├─ high_freq_var.yaml
│  ├─ low_freq_he.yaml
│  ├─ low_freq_hedging.yaml
│  ├─ low_freq_var.yaml
│  └─ node_info.json
├─ utils/
├─ create_all_h5.py
├─ create_h5.py
├─ dm.py
├─ GATsHAR.py
├─ high_freq_he.py
├─ high_freq_var.py
├─ low_freq_hedging.py
├─ low_freq_var.py
├─ stand_h5_train_val_test.py
├─ train_GNN.py
├─ train_single_final.py
└─ requirements.txt
```

## Main Components

### 1. Data preparation

- `create_h5.py`
  Build HDF5 feature files from raw data.
- `create_all_h5.py`
  Batch data construction script for multiple datasets.
- `stand_h5_train_val_test.py`
  Standardize train / validation / test splits separately and export statistics.

### 2. Model training

- `train_single_final.py`
  Main training script for the final GATs-HAR model.
- `train_GNN.py`
  Earlier / alternative GNN training script.
- `GATsHAR.py`
  Main model-related entry script used in this project version.

### 3. Ablation experiments

- `ablation/run_ablation_high.py`
  High-frequency ablation experiments.
- `ablation/run_ablation_low.py`
  Low-frequency ablation experiments.

### 4. Statistical and economic evaluation

- `dm.py`
  Diebold-Mariano significance test.
- `low_freq_var.py`
  Low-frequency VaR backtesting.
- `high_freq_var.py`
  High-frequency VaR backtesting.
- `low_freq_hedging.py`
  Low-frequency hedging effectiveness evaluation.
- `high_freq_he.py`
  High-frequency hedging effectiveness evaluation.

## Environment

Recommended:

- Python 3.10 or 3.11
- PyTorch with CUDA support if GPU is available

Install dependencies:

```bash
pip install -r requirements.txt
```

If you use GPU training, install the correct `torch` version for your CUDA environment before installing the remaining packages.

## Required Data

The data used in this project are obtained from the Wind database. Before running the code, you should organize the processed inputs locally and update the paths in the YAML configuration files under `config/`.

The exact file paths are controlled by:

- `config/GNN_param.yaml`
- `config/GNN_param_optuna.yaml`
- `config/ablation_full.yaml`
- `config/high_freq_he.yaml`
- `config/high_freq_var.yaml`
- `config/low_freq_hedging.yaml`
- `config/low_freq_var.yaml`

Before running any script, check these configuration files carefully.

## Usage

### 1. Build feature files

```bash
python create_h5.py
python create_all_h5.py
```

### 2. Standardize splits

```bash
python stand_h5_train_val_test.py
```

### 3. Train the main model

```bash
python train_single_final.py
```

If you want to use the alternative training pipeline:

```bash
python train_GNN.py
```

### 4. Run ablation experiments

High-frequency:

```bash
python ablation/run_ablation_high.py
```

Low-frequency:

```bash
python ablation/run_ablation_low.py
```

### 5. Run DM test

```bash
python dm.py
```

### 6. Run economic evaluation

Low-frequency VaR:

```bash
python low_freq_var.py
```

High-frequency VaR:

```bash
python high_freq_var.py
```

Low-frequency hedging effectiveness:

```bash
python low_freq_hedging.py
```

High-frequency hedging effectiveness:

```bash
python high_freq_he.py
```

## Configuration Notes

Most scripts read parameters from YAML files in `config/`.

Important settings include:

- sequence length
- train / validation split proportions
- node and edge feature composition
- training objective
- rolling window length
- VaR confidence level
- hedging correlation window

If results look inconsistent, first verify that:

- file paths in YAML are correct
- the dataset frequency matches the script
- the prediction files and realized series are aligned

## Output

Depending on the script, outputs may include:

- trained model checkpoints
- prediction files
- ablation summaries
- VaR backtesting tables
- hedging effectiveness summaries
- alignment and diagnostic CSV files

Output locations are controlled by the corresponding configuration files.