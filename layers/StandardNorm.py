import torch
import torch.nn as nn

class Normalize(nn.Module):
    def __init__(self, enc_in, affine=True, non_norm=False):
        super(Normalize, self).__init__()
        self.affine = affine
        self.non_norm = non_norm
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, enc_in))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, enc_in))

    def forward(self, x, mode='norm'):
        if self.non_norm:
            return x
        if mode == 'norm':
            return self._normalize(x)
        elif mode == 'denorm':
            return self._denormalize(x)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def _normalize(self, x):
        # Ensure x is 3D: (Batch, Time, Features)
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # Add feature dimension if missing
        elif x.dim() < 2 or x.dim() > 3:
            raise ValueError(f"Input tensor x must be 2D or 3D, but got shape {x.shape}")

        mean = x.mean(dim=1, keepdim=True)
        std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = (x - mean) / std

        if self.affine:
            # Adjust affine parameters to match the size of x dynamically
            if self.affine_weight.size(2) != x.size(2):
                self.affine_weight = nn.Parameter(torch.ones(1, 1, x.size(2)).to(x.device))
                self.affine_bias = nn.Parameter(torch.zeros(1, 1, x.size(2)).to(x.device))
            x = x * self.affine_weight + self.affine_bias

        return x

    def _denormalize(self, x):
        # Ensure x is 3D: (Batch, Time, Features)
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # Add feature dimension if missing
        elif x.dim() < 2 or x.dim() > 3:
            raise ValueError(f"Input tensor x must be 2D or 3D, but got shape {x.shape}")

        mean = x.mean(dim=1, keepdim=True)
        std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5)

        if self.affine:
            x = (x - self.affine_bias) / self.affine_weight
        x = x * std + mean

        return x
