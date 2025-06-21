import torch
import torch.nn as nn

from spatialConv import SpatialConvolution
import networkx as nx
import numpy as np
from utils.graph_utils import get_normalized_adjacency

# change name ~nadav (it's not spectral but spatial)
class DFTSpectralGCN(nn.Module):
    def __init__(self,
                 num_nodes,
                 time_samples,
                 in_features,
                 out_features,
                 G,                  # networkx graph
                 H,
                 num_classes,
                 hidden_dim = 64):
        super(DFTSpectralGCN, self).__init__()

        self.T = time_samples
        self.N = num_nodes
        self.G = out_features

        # get the normalized adjacency matrix
        A_tilde = get_normalized_adjacency(G)

        self.gcn = SpatialConvolution(time_samples,
            in_features,
            out_features,
            A_tilde,
            H)

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
    

    def forward(self, X):
        """
        X: shape (B, T, K, N)
           B = batch, T = time samples, K = input features per node, N = # nodes
        """

        X = X.to(self.device)  # ensure X is on the correct device

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
