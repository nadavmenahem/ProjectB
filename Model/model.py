import numpy as np
import os
import networkx as nx

# Root path for all datasets
DATA_ROOT = "outage_dataset"
# Choose the subfolder (e.g., "ieee39")
NETWORK_NAME = "ieee118"
GRAPH_FILE = "graph.npy"  # New filename for the graph

def load_dataset():
    """
    Load all case simulations for the selected network.
    """
    print("Loading dataset...")

    case_dir = os.path.join(DATA_ROOT, NETWORK_NAME, "cases")
    X, Y = [], []

    for filename in sorted(os.listdir(case_dir)):
        if filename.endswith(".npz"):
            data = np.load(os.path.join(case_dir, filename))
            X.append(data["x"])
            Y.append(data["y"])

    X = np.stack(X)  # shape: [num_cases, timesteps, num_buses]
    Y = np.stack(Y)  # shape: [num_cases, num_lines]

    return X, Y

def get_graph():
    """
    Load the graph topology from the topology.npy file.
    """
    path = os.path.join(DATA_ROOT, NETWORK_NAME, GRAPH_FILE)
    edge_index = np.load(path)

    edge_list = list(zip(edge_index[0], edge_index[1]))
    G = nx.Graph()
    G.add_edges_from(edge_list)

    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G


def main():
    G = get_graph()
    print("Graph loaded.")

    X, Y = load_dataset()
    print("Dataset loaded.")
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

if __name__ == "__main__":
    main()
