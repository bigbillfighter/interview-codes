import math
import numpy as np

import torch
from torch import nn 
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads, bias=False):
        super().__init__()
        self.dim = input_dim
        self.num_heads = num_heads
        self.split_dim = input_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.split_dim)

        self.linear_query = nn.Linear(input_dim, input_dim, bias=bias)
        self.linear_key = nn.Linear(input_dim, input_dim, bias=bias)
        self.linear_value = nn.Linear(input_dim, input_dim, bias=bias)
    
    def forward(self, query, key, value, mask=None):
        '''
        query: B, L, C
        '''
        B, L, _ = query.size()
        query = self.linear_query(query)
        key = self.linear_key(key)
        value = self.linear_value(value)

        query = query.reshape(B, L, self.num_heads, self.split_dim).permute(0, 2, 1, 3) # B, N, L, C//N
        key = key.reshape(B, L, self.num_heads, self.split_dim).permute(0, 2, 1, 3)
        value = value.reshape(B, L, self.num_heads, self.split_dim).permute(0, 2, 1, 3)

        attn = query * self.scale @ key.transpose(2, 3)

        if mask is not None:
            # mask: B, L
            mask = mask[:, None, None, :].repeat(1, self.num_heads, L, 1) # B, N, L, L
            attn = attn.masked_fill(torch.logical_not(mask), float("-inf"))

        attn = F.softmax(attn, dim=-1)

        out = attn @ value
        out = out.permute(0, 2, 1, 3).reshape(B, L, -1)

        return out


if __name__ == '__main__':
    attention = MultiHeadAttention(input_dim=4, num_heads=2)
    
    ## 输入
    x = torch.randn(2, 8, 4)
    mask = torch.zeros((2, 8), dtype=torch.float32)
    mask[0, :6] = 1.0
    mask[1, :3] = 1.0

    out = attention(x, x, x)
    print(out)

    out = attention(x, x, x, mask=mask)
    print(out)
