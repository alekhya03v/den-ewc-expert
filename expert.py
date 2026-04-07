# expert.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpandableExpert(nn.Module):
    def __init__(self, input_dim=784, h1=400, h2=400, num_classes=10):
        super().__init__()

        self.input_dim = input_dim
        self.h1_dim = h1
        self.h2_dim = h2
        self.num_classes = num_classes

        self.fc1 = nn.Linear(input_dim, h1)
        self.fc2 = nn.Linear(h1, h2)

        self.heads = nn.ModuleDict()

    def add_head(self, task_id, num_classes=10):
        key = str(task_id)
        if key not in self.heads:
            self.heads[key] = nn.Linear(self.h2_dim, num_classes)

    def forward_features(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def forward(self, x, task_id):
        feats = self.forward_features(x)
        logits = self.heads[str(task_id)](feats)
        return logits

    def expand_h2(self, n_new=64):
        """
        DEN-style simple expansion:
        Increase hidden layer 2 size from h2_dim -> h2_dim + n_new
        Then expand all heads accordingly.
        """
        old_fc2 = self.fc2
        old_h2 = self.h2_dim
        new_h2 = old_h2 + n_new

        new_fc2 = nn.Linear(self.h1_dim, new_h2)

        with torch.no_grad():
            new_fc2.weight[:old_h2] = old_fc2.weight
            new_fc2.bias[:old_h2] = old_fc2.bias

            nn.init.xavier_uniform_(new_fc2.weight[old_h2:])
            nn.init.zeros_(new_fc2.bias[old_h2:])

        self.fc2 = new_fc2
        self.h2_dim = new_h2

        # Expand all heads
        old_heads = self.heads
        new_heads = nn.ModuleDict()

        for task_key, old_head in old_heads.items():
            out_dim = old_head.out_features
            new_head = nn.Linear(new_h2, out_dim)

            with torch.no_grad():
                new_head.weight[:, :old_h2] = old_head.weight
                new_head.bias = old_head.bias

                nn.init.xavier_uniform_(new_head.weight[:, old_h2:])

            new_heads[task_key] = new_head

        self.heads = new_heads

    def get_model_size(self):
        return sum(p.numel() for p in self.parameters())
