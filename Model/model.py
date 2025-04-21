import torch
import torch.nn as nn

from spectralConv import SpectralConvolution


class SpectralGCN(nn.Module):
    def __init__(self, num_nodes, in_features, out_features, Lambda, H, num_classes):
        super(SpectralGCN, self).__init__()

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
            nn.Softmax(dim=1)
        )

    def forward(self, X_hat):  # shape: (B, T, K, N)
        out = self.gcn(X_hat)            # shape: (B, T, G, N)
        out = self.activation(out)
        return self.classifier(out)

