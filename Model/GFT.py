import networkx as nx
import numpy as np


def gft(G, signal):
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