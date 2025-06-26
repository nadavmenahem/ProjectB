import torch
import torch.nn as nn

from spectralConv import SpectralConvolution
from utils.graph_utils import graph_spectral_decomposition


class SpectralGCN(nn.Module):
    def __init__(self, num_nodes, time_samples, in_features, out_features, G, H, num_classes, hidden_dim = 32):
        super(SpectralGCN, self).__init__()

        Λ, V  = graph_spectral_decomposition(G)
        self.GFT    = torch.tensor(V.T, dtype=torch.float32)  # (N, N)
        self.Λ      = torch.tensor(Λ,    dtype=torch.float32)  # (N,)
        self.T      = time_samples
        self.K      = in_features
        self.G      = out_features
        self.N      = num_nodes

        self.gcn = SpectralConvolution(
            time_samples = self.T,
            in_features  = self.K,
            out_features = self.G,
            Lambda       = self.Λ,
            H            = H
        )

        self.activation = nn.ReLU()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_features * num_nodes, hidden_dim),
            nn.Dropout(p=0.3), # take from config ~nadav
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
            nn.Dropout(p=0.3)
        )


    @property
    def device(self):
        # grabs the device of the first parameter
        return next(self.parameters()).device


    def forward(self, X):               # X: (B, T, K, N)=(batch, time, features, nodes)
        X = X.to(self.device)

        # 1) Graph Fourier transform
        X_hat = torch.einsum("ij,btkj->btki", self.GFT, X)  # (B, T, K, N)

        # 2) Spectral conv → (B, G, N)
        out = self.gcn(X_hat)
        out = self.activation(out)      # still (B, G, N)

        # 3) Flatten (G,N) → (G*N)
        B = out.shape[0]
        out = out.reshape(B, self.G * self.N)  # (B, G*N)

        # 4) Classify
        return self.classifier(out)     # now matches (G*N → num_classes)


