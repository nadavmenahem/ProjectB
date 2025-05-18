import torch
import torch.nn as nn

class SpectralConvolution(nn.Module):
    def __init__(self, time_samples, in_features, out_features, Lambda, H):
        super(SpectralConvolution, self).__init__()
        self.T       = time_samples  # of time‐steps
        self.K       = in_features   # of node‐features
        self.G       = out_features  # of output filters
        self.H       = H             # polynomial order
        self.Λ       = Lambda        # (N,)-tensor of graph‐eigenvalues

        # now weights[t,k,g,h] is a distinct scalar
        self.weights = nn.Parameter(
            torch.randn(self.T, self.K, self.G, self.H + 1)
        )
        # remove ~nadav
        print("\n")

        print(f"number of weights: {self.weights.numel()}")
        print(f"weights shape: {self.weights.shape}")
        print(f"in features: {in_features}")
        print(f"out features: {out_features}")

        print("\n")
        # ~nadav

    def forward(self, X_hat):  
        # X_hat: (B, T, K, N)
        B, T, K, N = X_hat.shape
        assert T == self.T and K == self.K

        Y = torch.zeros(B, self.G, N, device=X_hat.device)

        # four nested loops: time, feature‐channel, output‐channel, poly‐order
        for t in range(T):
            for k in range(K):
                # slice out the (batch × nodes) at time t, feature k
                x_tk = X_hat[:, t, k, :]            # (B, N)

                for g in range(self.G):
                    conv_tkg = torch.zeros_like(x_tk)  # (B, N)

                    for h in range(self.H + 1):
                        # graph‐spectral multiplier λₙ^h
                        λh = (self.Λ ** h).view(1, N)     # (1, N)
                        # pick the scalar weight for (t,k,g,h)
                        w  = self.weights[t, k, g, h]    # scalar
                        # multiply node‐wise and accumulate
                        conv_tkg += w * (x_tk * λh)

                    # add this (t,k,g,*) contribution into the gth channel
                    Y[:, g, :] += conv_tkg

        return Y
