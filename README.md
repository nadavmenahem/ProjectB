# Power System Stability Analysis in the Presence of Line Outages using Graph Neural Networks

> End-to-end framework for **power‑grid line‑outage detection** on PMU data using **graph neural networks (GNNs)**, with **conformal prediction** for calibrated uncertainty and optional **Optuna** hyperparameter search.

![img.png](img.png)

---

## Table of Contents
- [Overview](#overview)
- [Repo Layout](#repo-layout)
- [Quick Start](#quick-start)
- [Data](#data)
- [Models & Architecture](#models--architecture)
- [Training & Evaluation](#training--evaluation)
- [Conformal Prediction (APS/RAPS)](#conformal-prediction-apsraps)
- [Hyperparameter Optimization (Optuna)](#hyperparameter-optimization-optuna)
- [Results](#results)
- [Complexity & Runtime](#complexity--runtime)
- [Reproducibility](#reproducibility)
- [References](#references)

---

## Overview

Power‑grid reliability hinges on early **line‑outage detection**, a task made harder by the variability and scale of modern, renewables‑rich networks. This project takes a **data‑driven** route: we model the grid as a **graph** (buses = nodes, lines = edges), use high‑rate **PMU phasors**, and learn the spatio‑temporal signatures of outages. Building on **He & Cheng (ICPR 2021)**, we implement two GCN variants—**Spatial** (graph‑shift polynomials) and **Spectral/GFT** (graph‑frequency filtering)—and compare them on the **IEEE‑39** system. The models classify both **single** and **simultaneous** line outages, scaling naturally to **N–k** contingencies without explicit system parameters.

Beyond the architectures, the repo includes an end‑to‑end pipeline for **reproducible simulation & data generation**, training/evaluation, and **conformal prediction (APS/RAPS)** to produce **calibrated prediction sets** with reported **coverage** and **average set size**.

---

## Repo Layout

- `Model/` — GNN architectures (**Spectral GCN / GFT** and **Spatial GCN**), train/eval runners, **conformal prediction (APS/RAPS)** utils, configs, and plotting.
- `Simulations/` — IEEE test‑system simulators (Pandapower), outage scenario generation (N, N‑1, N‑k), optional noise injection, and export to `datasets/`.
- `helpers/` — small helpful scripts (dataset checks, splits, ...).
- `datasets/` — generated datasets organized by grid (e.g., `ieee39/`).
- `checkpoints/` — saved model weights and artifacts from hyperparameter searches (Optuna trials).
- `projectReport.pdf` — paper‑style report with methods, experiments, and CP‑calibrated results.
- `finalPresentation.pdf` — slide deck overview (motivation → method → results), aligned with the report.

> The `datasets/` and `checkpoints/` folders are created on demand by the scripts.

---

## Data

### Quick generation:
1) **Configure data settings**

Edit `Simulations/PandaPower/data_config.yaml` to set timing, noise, output path, etc. For example:
   
```yaml
# Configuration file for the generate_noisy_data.py script (and not for generate_data.py)

TOTAL_TIME:     40  # seconds
SAMPLING_RATE:  1   # Hz
OUTAGE_TIME:    20  # if None, generate_noisy_data2.py will generate random outage times
DATA_ROOT:      "datasets/TT40_OT20_NS003_M40_N-2"  # Path to save the dataset
NOISE_SCALE:    0.03  # Scale of noise to be added to loads
MULTIPLICITY:   40    # Number of cases to generate for each line outage
N2_CONTINGENCY: True  # N-1 or N-2 contingency
```
* Cases with the same outage scenario differ only in the random noise applied to the loads.
2) **Run the generator**:
```bash
python Simulations/PandaPower/generate_noisy_data.py --config Simulations/PandaPower/data_config.yaml
```
3) **Select the test system at the prompt**
```text
Enter number of buses (e.g., '39' for IEEE 39):
```
4) **Output**
- Cases → `<DATA_ROOT>/<network_name>/cases/case_XXXX.npz` (signals, labels)
- Topology → `<DATA_ROOT>/<network_name>/graph.npy` (`[2, E]` edge index)
- Dataset metadata → `<DATA_ROOT>/<network_name>/meta.json`
- **Outage times** → `<DATA_ROOT>/<network_name>/outage_times.json` 
  (only created when outage times are randomized; records the actual outage step chosen for each case).

### Noise model (load-driven, physics-consistent)
- Variability is injected **on the loads/injections**, **not** by corrupting the PMU signals post hoc.  
- **Replicas differ only by the random noise draws**, the outage pattern and base topology are identical across replicas of the same scenario.

### Step-wise (not a dynamic simulation)
- We run a fixed number of **independent steady-state solves** per case (e.g., 40 steps).  
- An outage is applied at a designated **step index** (e.g., step 20). Steps are **exchangeable** snapshots; the index merely stands in for “time.” No transient/differential dynamics are simulated.

### Feature signals (per node)
- **Primary signal (used in this work):** bus voltage **phase angles** ($\theta$) taken from each step’s **steady-state power-flow** solution (after the load perturbation). 
- **Extensible inputs (optional):** the models also accept additional per-bus features—e.g., absolute voltage magnitudes \(|V|\)

### Graph construction
- **Nodes** = buses; **Edges** = transmission lines.  
- A graph shift (adjacency/Laplacian) is formed once for the base system and reused across replicas.
### example
example of measured data on bus 32 for a scenario of IEEE39 where lines 7 and 27 are tripped: 

![img_4.png](img_4.png)

---

## Training the Model

### Models & Architecture

Two main variants following **He & Cheng (ICPR 2021)**:

1. **Spectral GCN (GFT)**  
   - **Preprocessing:** compute the **graph Fourier basis** (Laplacian eigenvectors).  
   - **Convolution:** apply **polynomial spectral filters**
     $w(\Lambda) = \sum_{h=0}^{H} w_h \Lambda^h$.  
   - **Classifier:** linear layers operating in the **graph-frequency** domain.

2. **Spatial GCN (DFT + Graph-shift)**  
   - **Preprocessing:** apply a DFT-style transform to input signals.  
   - **Convolution:** use **polynomials of a graph-shift operator** (e.g., normalized adjacency or Laplacian).  
   - **Classifier:** same as in the spectral GCN.


### Quick training:
1) **Configure the model**

Edit `Model/configs/config.yaml` to set dataset paths, split ratios, model type, training hyperparameters, conformal prediction settings, etc. For example:
   
```yaml
# === Dataset Settings ===
dataset:
  path: "datasets/TT40_OT20_NS003_M40_N-2"
  network_name: "ieee39"  # Change this to the desired network name
  graph_path: "graph.npy"
  metadata_path: "meta.json"
  test_size: 0.1    # Fraction of data to be used for testing
  val_size:  0.1    # For validation
  cal_size : 0.1    # For calibration

# === Model Settings ===
model:
  load_pretrained: True # Load pretrained model weights
  type: "gft"           # Specify the model type ("dft", "gft")
  params_folder: "Model/params"
  hidden_dim: 64

# === Spectral Convolution ===
output_features: 20   # G: Number of spectral filters (output features)
poly_order: 2         # H: Degree of the frequency-domain polynomial filter

# === Training ===
batch_size: 64
num_epochs: 10
learning_rate: 0.0005
epsilon: 0  # smoothing the labels - CURRENTLY NOT USED

# === Regularization ===
dropout: 0.0

# === Conformal Prediction ===
alpha: 0.1  # Significance level for conformal prediction
topk: 3     # Number of top predictions to consider
  
# === Optimization ===
optimizer: "adam"     # Optimizer type (e.g., "adam", "sgd")
loss_function: "dkl"  # Support for "bce" (binary cross-entropy), "dkl" (Kullback-Leibler divergence) and "Jeffreys" (Jeffrey divergence)
entropy_weight: 0     # Weight for entropy term in the DKL loss function
scheduler: "cosine"   # Supports "ReduceLROnPlateau", "Cosine", "steplr"
step_size: 2          # StepLR arg
gamma: 0.6            # StepLR arg
```
2) **Hyper-parameter search using Optuna:** (optional, if not, skip to step 4)
```bash
python Model/tune_model.py --config Model/configs/config.yaml
```
- The script defines which parameters to search and how many trials to run.
- Results are logged to `optuna.db`.
- To explore trials visually: 
   ```bash
   optuna-dashboard sqlite:///optuna.db --port 8080
   ```
  Open http://localhost:8080 in your browser
3) **Export the best trial to a config file**
```bash
python helpers/build_best_config.py 
```
Edit the script to point to your Optuna study:
```python
STUDY_NAME = "spectral_gcn_tuning_20250903_111311"   # copy from dashboard title
OUT_YAML   = REPO_ROOT / "Model" / "configs" / "best.yaml"
```
4) **Train the model**:
```bash
python Model/main.py --config Model/configs/config.yaml
# or, to use the tuned config:
python Model/main.py --config Model/configs/best.yaml
```
> Set `RESULT_PLOTTING = True` in `main.py` to produce visualizations of random inference examples during training.

Example of Optuna search: 
![img_3.png](img_3.png)

---

## Conformal Prediction (APS/RAPS)

We use **conformal prediction (CP)** to convert raw model scores into **prediction sets** with statistical coverage guarantees.  
Practically, you’ll see two numbers alongside accuracy: **coverage** (how often the true outage is inside the set) and **average set size**.

- **APS**: builds a set by accumulating classes until a calibrated threshold is reached.  
- **RAPS**: adds a penalty to low-ranked classes, often yielding **smaller sets** at the same coverage.

**In this repo:** CP runs **automatically inside `Model/main.py`** whenever you define a **calibration split**.  
Just set these in your config:
- `cal_size` > 0 → hold out data for CP calibration  
- `alpha` → target risk level (e.g., `0.1` ≙ 90% coverage)  

The CP **method** (APS or RAPS) is selected in `main.py`. Current version uses RAPS.

**Outputs:** CP metrics (coverage, avg set size) are logged with the usual evaluation results and saved under your run/output directory.  
No separate command is needed.

---

## Results

- **Top‑3** (test): `92.79%`  
- **CP (APS/RAPS)** at ($\alpha=0.1$): coverage = `95.81%`, avg set size = `3.42`  
- **Best architecture:** the strongest results were achieved with the **GFT model** using a **polynomial of order 1** and **two fully connected layers**.

**Examples of result visualization on the IEEE-39 system.**  
Each bar plot shows the model’s **predicted outage probabilities** (sorted by score), with true outages marked by a star $\star$.  
- In **Case 46**, one of the true outaged lines ranks low in raw probability and falls outside the top-3 predictions. However, it is still captured in the **conformal prediction set**, demonstrating how CP provides coverage even when standard top-k accuracy fails.  
- In **Case 1751**, all true outages are ranked near the top and included in both the top predictions and the conformal set, leading to small set size and high confidence.

![img_1.png](img_1.png)
![img_2.png](img_2.png)

---

## Complexity & Runtime

- Forward pass of the GCN scales roughly as **O(N²)** for \(N\) buses.  
- **Complexity & runtime:** a GCN forward pass scales as $O(N^2)$ and takes only ~0.7 ms per case (on CPU), compared to classical methods at $O(N^{3+k})$; memory footprint is modest (per-batch scaling with node count).


---

## References

- J. He and M. Cheng, “Graph Convolutional Neural Networks for Power Line Outage Identification,” *Proc. ICPR*, 2021.  
**If you use this repo**, please cite the paper above.
- Anastasios N. Angelopoulos and Stephen Bates, "A Gentle Introduction to Conformal Prediction and Distribution-Free 
Uncertainty Quantification."
- nastasios N. Angelopoulos and Stephen Bates, "Uncertainty Sets for Image Classifiers using Conformal Prediction."