import torch
from torch import nn


class BatchNormlization(nn.Module):
    def __init__(self, n_channel, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.randn(n_channel))
        self.beta = nn.Parameter(torch.randn(n_channel))
        self.eps = eps
    
    def forward(self, x):
        """
        x: b, c, h, w
        """
        assert len(x.size()) == 4
        B, C, H, W = x.shape
        mu = x.reshape(B, -1).mean(dim=1)[:, None, None, None]
        var = x.reshape(B, -1).var(dim=1, unbiased=False)[:, None, None, None]

        x_hat = (x - mu) / torch.sqrt(var + self.eps)
        x = x_hat * self.gamma[None, :, None, None] + self.beta[None, :, None, None]
        return x

if __name__ == "__main__":
    input_data = torch.randn(4, 2, 10, 8)
    model = BatchNormlization(2)
    output = model(input_data)
    print(output)