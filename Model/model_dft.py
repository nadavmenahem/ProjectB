import torch
import torch.nn as nn

from spectralConv import SpectralConvolution
from graph_utils import graph_spectral_decomposition

class DFTSpectralGCN(nn.Module):
    def __init__(self,
                 num_nodes: int,
                 time_samples: int,
                 in_features: int,
                 out_features: int,
                 G,                  # networkx graph
                 H: int,
                 num_classes: int):
        super(DFTSpectralGCN, self).__init__()

        # we only need the eigenvalues for the spectral convolution
        Lambda, _ = graph_spectral_decomposition(G)
        self.Lambda = torch.tensor(Lambda, dtype=torch.float32)  # (N,)

        self.T = time_samples
        self.N = num_nodes
        self.G = out_features

        # build your spectral‐conv (over the graph) exactly as before
        self.gcn = SpectralConvolution(
            time_samples = self.T, # num of time samples is the same as num of frequencies
            in_features  = in_features,
            out_features = out_features,
            Lambda       = self.Lambda,
            H            = H
        )

        self.activation = nn.ReLU()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_features * num_nodes, num_classes),
        )

    def forward(self, X):
        """
        X: shape (B, T, K, N)
           B = batch, T = time samples, K = input features per node, N = # nodes
        """

        # 1) DFT along time axis → complex tensor of shape (B, F, K, N),
        #    where F = T//2 + 1 frequency bins.
        Xf = torch.fft.fft(X, dim=1)
        # print(f"DFT shape: {type(Xf)}, shape: {Xf.shape} (B, F, K, N)")

        # 2) take magnitude → real tensor (B, F, K, N)
        Xmag = torch.abs(Xf)

        # 3) graph‐spectral convolution (independent at each freq & feature)
        #    → out shape (B, G, N)
        out = self.gcn(Xmag)

        # 4) nonlinearity
        out = self.activation(out)

        # 5) collapse freq dimension by average pooling
        #    → (B, G, N)
        # out = out.mean(dim=1)

        # 6) flatten (G,N) → vector of length G*N
        B = out.shape[0]
        out = out.view(B, self.G * self.N)

        # 7) final classifier
        return self.classifier(out)
