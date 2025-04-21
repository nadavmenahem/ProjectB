import networkx as nx
import numpy as np
from scipy import sparse


def graph_spectral_decomposition(G):
    # Get laplacian matrix
    # L = nx.laplacian_matrix(G).toarray()

    # Get adjacency matrix
    A = nx.adjacency_matrix(G).toarray()  # Convert sparse matrix to dense NumPy array

    # Compute the degree matrix
    degrees = np.array(A.sum(axis=1)).flatten()
    D_inv_sqrt = sparse.diags(1.0 / np.sqrt(degrees))

    # 2. Compute eigenvalues and eigenvectors (Graph Fourier basis)
    eigenvalues, eigenvectors = np.linalg.eigh(A)

    return eigenvalues, eigenvectors
