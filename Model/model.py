import torch
import torch.nn as nn

from spectralConv import SpectralConvolution
from graph_utils import graph_spectral_decomposition


class SpectralGCN(nn.Module):
    def __init__(self, num_nodes, in_features, out_features, G, H, num_classes):
        super(SpectralGCN, self).__init__()

        Lambda, GFTT  = graph_spectral_decomposition(G)
        self.GFT = torch.tensor(GFTT.T, dtype=torch.float32)  # torch tensors
        Lambda = torch.tensor(Lambda, dtype=torch.float32)

        self.gcn = SpectralConvolution(
            in_features=in_features,
            out_features=out_features,
            Lambda=Lambda,
            H=H
        )

        self.activation = nn.ReLU()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_features * num_nodes, num_classes),
            # nn.Softmax(dim=1)
        )

    def forward(self, X):  # shape: (B, T, K, N) = batch, time, features, nodes
        X_hat = torch.einsum("ij,btkj->btki", self.GFT, X)  # GFT transform on the last dim (for every sample in the batch, for every time step and for every feature)
        # X_hat = X  # no GFT ~nadav 
        out = self.gcn(X_hat)  # shape: (B, T, G, N)
        out = self.activation(out)
        out = out.mean(dim=1)  # average over time dimension (T)
        return self.classifier(out)

