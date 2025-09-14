# spatialConv.py  (new file -- drop alongside spectralConv.py)
import torch, torch.nn as nn

class SpatialConvolution(nn.Module):
    """
    Implements   Y = Σ_h  W_h · (Â^h X)   with learnable W_h.
    X shape : (B, T, K, N)
    Y shape : (B, G, N)
    """
    def __init__(self, T, K, G, A_tilde, H):
        super().__init__()
        self.T, self.K, self.G, self.H = T, K, G, H
        self.register_buffer("A_tilde", torch.tensor(A_tilde, dtype=torch.float32))
        # learnable coeffs: (T, K, G, H+1)
        self.coeffs = nn.Parameter(torch.randn(T, K, G, H+1))

    def forward(self, X):
        # X: (B, T, K, N)
        B, T, K, N = X.shape
        Y = torch.zeros(B, self.G, N, device=X.device)

        # we'll keep a running "power" of A_tilde times X
        # start with A^0 X = X itself
        X_h = X                                # shape (B, T, K, N)

        for h in range(self.H + 1):
            # coeffs for this hop h: shape (T, K, G)
            w_h = self.coeffs[..., h]          # (T, K, G)

            # accumulate into Y via einstein summation:
            #   X_h: (B, T, K, N)
            #   w_h: (    T, K, G)
            # want: sum_{t,k} X_h[b,t,k,n] * w_h[t,k,g]
            # result: (B, G, N)
            Y = Y + torch.einsum("btkn,tkg->bgn", X_h, w_h)

            # now step X_h ← A_tilde · X_h  for next power
            # (just matrix-multiply the N-dim slice)
            X_h = torch.einsum("ij,btkj->btki", self.A_tilde, X_h)

        return Y  # (B, G, N)