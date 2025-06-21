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

    # Sort eigenvalues and eigenvectors
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors


def get_normalized_adjacency(G):
    """
    Compute the normalized adjacency matrix of a graph G.
    Returns the normalized adjacency matrix as a NumPy array.
    """
    A = nx.adjacency_matrix(G).astype(np.float32).todense()  # Convert to dense format
    degrees = np.array(A.sum(axis=1)).flatten()
    D_inv_sqrt = sparse.diags(1.0 / np.sqrt(degrees, where=degrees > 0))
    A_normalized = D_inv_sqrt @ A @ D_inv_sqrt
    return A_normalized