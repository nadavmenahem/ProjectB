import numpy as np
import os
import networkx as nx


DATA_PATH = "outage_dataset"  # Path to the dataset
GRAPH = "Graph" # name of file to load graph from


def load_dataset():
    """
    Load the dataset from the specified directory.
    Each file contains power flow data for a specific case.
    """
    print("Loading dataset...")

    X = []
    Y = []

    for filename in sorted(os.listdir(DATA_PATH)):
        if filename.endswith(".npz"):
            data = np.load(os.path.join(DATA_PATH, filename))
            X.append(data["x"])  # shape: [1600, num_buses]
            Y.append(data["y"])  # shape: [num_lines]

    # Convert lists to arrays
    X = np.stack(X)  # shape: [num_cases, 1600, num_buses]
    Y = np.stack(Y)  # shape: [num_cases, num_lines]

    return X, Y

def get_graph():
    # Load the edge index
    edge_index = np.load(os.path.join(DATA_PATH, f"{GRAPH}.npy"))  # shape: [2, num_edges]

    # Transpose to list of edges
    edge_list = list(zip(edge_index[0], edge_index[1]))

    # Create the undirected graph in NetworkX
    G = nx.Graph()
    G.add_edges_from(edge_list)

    # Done! G is your graph
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    return G



def main():

    G = get_graph()  # Load the graph
    print("Graph loaded")

    X, Y = load_dataset()

    print("X shape:", X.shape)  # for IEEE 39-bus system: (20, 1600, 39)
    print("Y shape:", Y.shape)  # for IEEE 39-bus system: (20, 35)


if __name__=="__main__":
    main()