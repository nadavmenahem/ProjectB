import torch
import torch.nn as nn

class SpectralConvolution(nn.Module):
    def __init__(self, time_samples, in_features, out_features, Lambda, H):
        super().__init__()
        self.T, self.K, self.G, self.H = time_samples, in_features, out_features, H
        # Λ is a 1-D tensor of shape (N,)
        self.register_buffer('Lambda', Lambda)

        # precompute Λ^0…Λ^H → shape (H+1, N)
        h_idx = torch.arange(H+1, device=Lambda.device).unsqueeze(1)  # (H+1,1)
        self.register_buffer('Lambda_powers',
                             (Lambda.unsqueeze(0) ** h_idx))          # (H+1, N)

        # learnable coefficients w[t,k,g,h]
        self.weights = nn.Parameter(torch.randn(self.T,
                                                self.K,
                                                self.G,
                                                self.H+1))

    def forward(self, X_hat):
        """
        X_hat: (B, T, K, N)
        returns Y: (B, G, N)
        """
        B, T, K, N = X_hat.shape
        assert T == self.T and K == self.K and N == self.Lambda.size(0)

        # 1) lift X_hat into (H+1, B, T, K, N) by multiplying each hop:
        #    X_lp[h, b,t,k,n] = X_hat[b,t,k,n] * (Λ[n]**h)
        X_lp = X_hat.unsqueeze(0) * self.Lambda_powers.view(self.H+1, 1, 1, 1, N)

        # 2) reorder weights to align hops first: (H+1, T, K, G)
        w = self.weights.permute(3, 0, 1, 2)

        # 3) einsum sums over h, t, k:
        #    Y[b,g,n] = Σ_{h,t,k}  X_lp[h,b,t,k,n] * w[h,t,k,g]
        Y = torch.einsum('hbtkn,htkg->bgn', X_lp, w)

        return Y
