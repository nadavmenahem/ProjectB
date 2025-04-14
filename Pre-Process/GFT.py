import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


# Display results
# print("Original Signal: ", signal)
# print("GFT: ", gft)
# print("Reconstructed Signal: ", reconstructed_signal)


# plt.figure(figsize=(10, 4))

# plt.subplot(1, 2, 1)
# plt.stem(signal)
# plt.title("Original Signal on Graph")
# plt.xlabel("Node")
# plt.ylabel("Value")

# plt.subplot(1, 2, 2)
# plt.stem(gft)
# plt.title("Graph Fourier Transform")
# plt.xlabel("Frequency Index")
# plt.ylabel("Amplitude")

# plt.tight_layout()
# plt.show()


def GFT(G, signal):
    # Get laplacian matrix
    L = nx.laplacian_matrix(G).toarray()

    # Get adjacency matrix
    #A = nx.adjacency_matrix(G).toarray()  # Convert sparse matrix to dense NumPy array

    # 2. Compute eigenvalues and eigenvectors (Graph Fourier basis)
    eigenvalues, eigenvectors = np.linalg.eigh(L)

    # 4. Compute Graph Fourier Transform (GFT)
    gft = eigenvectors.T @ signal

    # 5. Inverse Graph Fourier Transform (reconstruct signal)
    reconstructed_signal = eigenvectors @ gft

    return gft



def visualize_graph(G, signal):

    # Get layout for consistent node positions
    pos = nx.spring_layout(G, seed=42)

    # Draw graph structure (nodes and edges)
    nx.draw_networkx_edges(G, pos, edge_color='red', style='dotted', alpha=0.7)
    nx.draw_networkx_nodes(G, pos, node_color='red', node_size=100)

    # Plot signal as vertical lines (bar-style)
    for i, (x, y) in pos.items():
        plt.plot([x, x], [y, y + signal[i]], color='blue', linewidth=2)

    plt.axis('off')
    plt.show()


def main():
    
    # Create a graph
    G = nx.path_graph(5)  # A simple path graph with 5 nodes
    G.add_edges_from([(0, 3), (1, 4)])  # Add edges between nodes

    # Get a graph
    # G, n = get_graph() # get graph and number of nodes

    # Create graph signal
    signal = np.array([1, 2, 3, 4, 5])

    # Get graph signal
    # signal = get_signal() # get a graph signal
        
    gft_sig = GFT(G, signal)
    print("GFT sig: ", gft_sig)

    visualize_graph(G, signal)
    visualize_graph(G, gft_sig)


if __name__ == "__main__":
    main()

