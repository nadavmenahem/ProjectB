# spatialConv.py  (new file -- drop alongside spectralConv.py)
import torch, torch.nn as nn
import numpy as np

class SpatialConvolution(nn.Module):
    """
    Implements   Y = Σ_h  W_h · (Â^h X)   with learnable W_h.
    X shape : (B, T, K, N)
    Y shape : (B, G, N)
    """
    def __init__(self, time_samples, in_feats, out_feats, A_tilde, H):
        super().__init__()
        self.T, self.K, self.G, self.H = time_samples, in_feats, out_feats, H

        # store Â⁰…Âᴴ  as (H+1, N, N) tensor
        A_powers = [np.eye(A_tilde.shape[0], dtype=np.float32)]
        for h in range(1, H + 1):
            A_powers.append(A_powers[-1] @ A_tilde)
        self.register_buffer("A_powers",
                             torch.tensor(np.stack(A_powers)))  # (H+1,N,N)

        # one scalar weight per (t, k, g, h)
        self.weights = nn.Parameter(
            torch.randn(time_samples, in_feats, out_feats, H + 1))

    def forward(self, X):
        B, T, K, N = X.shape
        Y = torch.zeros(B, self.G, N, device=X.device)

        for t in range(T):
            for k in range(K):
                x_tk = X[:, t, k, :]                        # (B,N)
                for g in range(self.G):
                    y_tkg = 0
                    for h in range(self.H + 1):
                        Ah = self.A_powers[h]               # (N,N)
                        w  = self.weights[t, k, g, h]       # scalar
                        y_tkg += w * (x_tk @ Ah.T)          # (B,N)
                    Y[:, g, :] += y_tkg
        return Y
