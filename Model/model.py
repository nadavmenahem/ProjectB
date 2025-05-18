import torch
import torch.nn as nn

from spectralConv import SpectralConvolution
from graph_utils import graph_spectral_decomposition


class SpectralGCN(nn.Module):
    def __init__(self, num_nodes, time_samples, in_features, out_features, G, H, num_classes):
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
            nn.Linear(self.G * self.N, num_classes),
            # nn.Softmax(dim=1)
        )

    def forward(self, X):               # X: (B, T, K, N)
        # 1) Graph Fourier transform
        X_hat = torch.einsum("ij,btkj->btki", self.GFT, X)  # (B, T, K, N)

        # 2) Spectral conv → (B, G, N)
        out = self.gcn(X_hat)
        out = self.activation(out)      # still (B, G, N)

        # 3) Flatten (G,N) → (G*N)
        B = out.shape[0]
        out = out.view(B, self.G * self.N)  # (B, G*N)

        # 4) Classify
        return self.classifier(out)     # now matches (G*N → num_classes)


