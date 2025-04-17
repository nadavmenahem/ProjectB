# ⚡ Outage Simulation Dataset Generator for Power Systems

This tool generates simulation data for line outage scenarios in IEEE power system test cases. It's designed for researchers and developers working with graph-based models (e.g. GNNs) in the context of power system monitoring or outage detection.

---

## 🚀 How to Use

### 1. Run the script

```bash
python generate_data2.py
```

### 2. Choose a network

You'll be prompted to enter a number corresponding to an IEEE network.

**Examples:**

- `9` → IEEE 9-bus system
- `14` → IEEE 14-bus system
- `24` → IEEE 24-bus (RTS) system
- `30` → IEEE 30-bus system
- `39` → IEEE 39-bus system
- `118` → IEEE 118-bus system


---

## 💾 Dataset Structure

Each dataset is stored in a dedicated folder under `outage_dataset/` named after the selected network.

### Example: `outage_dataset/case39/`

```
outage_dataset/
└── ieee39/
    ├── topology.npy           # Bus connectivity as edge index (2 x num_edges)
    ├── meta.json              # Metadata (num_buses, num_lines, etc.)
    └── cases/
        ├── case_000.npz       # No outage
        ├── case_001.npz       # Single line outage
        ├── case_002.npz       # ...
        └── case_019.npz       # Double line outage
```

Each `.npz` file contains:

- `x`: ndarray of shape `[timesteps, num_buses]` — phasor angles over time  
- `y`: ndarray of shape `[num_lines]` — binary outage vector (1 = outaged line)

---

## 🌐 Graph Format

The graph is saved in `Graph.npy` as a NumPy array of shape `(2, num_edges)`, representing undirected edges between buses.

To load the graph with `networkx`:

```python
import numpy as np
import networkx as nx

edges = np.load("outage_dataset/case39/Graph.npy")
edge_list = list(zip(edges[0], edges[1]))

G = nx.Graph()
G.add_edges_from(edge_list)

print(f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
```

---

## ⚙️ Requirements

Install required libraries with:

```bash
pip install pandapower numpy pandas matplotlib networkx
```

---

## ✨ Notes

- Simulated outages are applied at `t = 100s`.
- Phasor angles are sampled at `8 Hz` for `200s` → 1600 timesteps.
- Both single and double line outages are included, plus a "no outage" baseline.
- Line and bus indices follow `pandapower` conventions.
