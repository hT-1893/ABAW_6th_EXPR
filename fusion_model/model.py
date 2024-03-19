import torch
import torch.nn as nn

class Fusion(nn.Module):

    def __init__(self, n_classes, dim1=768, dim2=768, hidden_dim=1024):
        super().__init__()
        
        self.mha = nn.MultiheadAttention(1, 1, batch_first=True)

        self.linear1 = nn.Linear(dim1, hidden_dim)
        self.linear2 = nn.Linear(dim2, hidden_dim)

        self.linear_key = nn.Linear(hidden_dim * 2, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Linear(hidden_dim // 2, n_classes)
        )
    
    def forward(self, x1, x2):

        x1 = self.linear1(x1)
        x2 = self.linear2(x2)

        key = torch.cat((x1, x2), dim=-1)
        key = self.linear_key(key)

        x1 = x1.unsqueeze(-1)
        x2 = x2.unsqueeze(-1)
        key = key.unsqueeze(-1)

        x, _ = self.mha(x2, key, x1)

        x = x.squeeze(-1)
        x1 = x1.squeeze(-1)

        x = x + x1

        x = self.mlp(x)

        return x