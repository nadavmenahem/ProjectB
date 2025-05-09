# 🧠 Spectral GCN for Power Line Outage Detection

This module implements the **Spectral Graph Convolutional Network (Spectral GCN)** for classifying time-evolving graph signals, specifically aimed at detecting power line outages from phasor measurements.

## 🏗 Architecture Overview

The model is based on the paper **"Graph Convolutional Neural Networks for Power Line Outage Identification"**. It uses **Graph Fourier Transform (GFT)** and frequency-domain convolution for learning from spatiotemporal data on power grids.

### Key Components

- `SpectralConvolution`: Custom PyTorch layer performing spectral convolution using polynomial filters in the graph frequency domain.
- `SpectralGCN`: Full model combining GFT, spectral convolution, activation, temporal pooling, and a linear classifier.
- `graph_spectral_decomposition`: Computes eigenvalues and eigenvectors for constructing the GFT basis from the graph adjacency matrix.

## 🧮 Input Format

- **X**: Tensor of shape `(B, T, K, N)`  
  - B = batch size  
  - T = time steps  
  - K = input features per node  
  - N = number of nodes
- **Y**: Binary vector of shape `(B, L)`  
  - L = number of power lines (edges in the graph)

## 🔄 Forward Pass

1. **GFT**: Applies graph Fourier transform to node signals.
2. **Spectral Convolution**: Applies trainable polynomial filters to frequency-domain signal.
3. **Activation**: Applies ReLU (configurable).
4. **Temporal Pooling**: Averages across time steps.
5. **Classification**: Outputs outage probabilities over all lines.

## ⚙️ Configurations

Set via `config.yaml`:

```yaml
output_features: 20     # G: output features
poly_order: 1           # H: polynomial degree
batch_size: 6
num_epochs: 25
learning_rate: 0.01
```

## 🧪 Running the Model

From the root directory:

```bash
python main.py
```

This will:
- Load the dataset
- Build and train `SpectralGCN`
- Evaluate its precision/recall
- Optionally plot predictions per test case

## 📈 Visualizations

- Use `plot_test_case_probs()` to visualize predicted vs. actual outage lines.
- Enable `DATA_PLOTTING` in `main.py` to visualize phasor angles over time.

## 🧠 Paper Reference

Jia He, Maggie Cheng. *"Graph Convolutional Neural Networks for Power Line Outage Identification"*, ICPR 2020.
