# Spectral GCN for Line‑Outage Detection (with Conformal Prediction)

A compact, end‑to‑end implementation of **line‑outage detection on power grids** 
using a **Spectral GCN** that operates in the **Spectral domain** (using GFT
as a preproccesing method). The repo also includes a **spatial GCN** (DFT as
preprocces), utilities for data simulation, and **conformal prediction 
(APS/RAPS)** for calibrated prediction sets.

> This is a high‑level overview. Folder‑level READMEs will cover details (data
> formats, hyper‑parameters, plots, etc.). <!-- ~nadav -->

---

## What’s inside
- **Data generation** (Pandapower): scripts to simulate IEEE‑style networks and create labeled outage datasets.
- **Models**
  - **Spectral GCN** — GFT preprocess then spectral graph convolution.
  - **Spatial  GCN** — DFT preprocess then spatial graph convolution.
- **Training & evaluation**: runners, metrics (Top‑k, loss curves), visualizations.
- **Conformal prediction**: APS/RAPS calibration → coverage + set size.
- **Tuning**: Optuna sweeps.
- **Configs & params**: YAML configs and (optionally) pretrained weights.

---

## Repo layout (top‑level)
- `Model/` — model definitions (Spectral GCN, DFT+Spatial), train/eval runners, CP utilities, configs, plotting.
- `Simulations/` — data simulators for IEEE grids (Pandapower) and helpers.
- `datasets/` — generated datasets live here.
- `helpers/` — small helpfull scripts.

> See each folder’s README (coming next) for exact file names and CLI examples. <!-- ~nadav -->

---

## Quick start (minimal)
1. **Install** (Python 3.10+): `torch`, `numpy`, `scipy`, `networkx`, `pandas`, `matplotlib`, `pyyaml`, `python‑box`, `optuna`, `pandapower`.
2. **Prepare data** using the simulators in `Simulations/` (see that folder’s README for arguments and presets).
3. **Train & evaluate** the chosen architecture via `Model/main.py` with your `Model/configs/config.yaml` (model type, paths, splits, α for CP, etc.).

---

## Typical workflow
1. Simulate/collect data → `datasets/.../ieeeXX/`.
2. Pick **model** = `spectral` (GFT) or `dft_spatial` in the config; set hidden dims / polynomial order *H* as needed.
3. Train or load weights → evaluate Top‑k and loss.
4. **Calibrate CP** (APS/RAPS) on a held‑out calibration split → report **coverage** and **average set size**.
5. Save plots and example cases for the report.

---

## Reproducibility notes
- Use fixed **random seeds** and stable **case‑wise splits** (train/val/cal/test) defined in the config.
- Keep the **calibration set** disjoint from training and testing to preserve CP guarantees.
- Store run configs and metrics in a lightweight log dir; Optuna DB is included for sweeps.

---

## Background
This project follows a spectral‑GCN approach (filtering in graph frequency space via GFT) and augments predictions with **conformal prediction** (APS/RAPS) to provide **statistical coverage** guarantees on prediction sets. See the `paper/` folder and sub‑READMEs for references and pointers.

---

## License
Add a license file (e.g., `LICENSE`, MIT or similar).

## Acknowledgments
- **Pandapower** for power‑system simulation.
- **Optuna** for hyper‑parameter optimization.
- Conformal prediction ideas inspired by recent APS/RAPS literature.
