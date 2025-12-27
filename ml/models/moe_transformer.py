import torch
import torch.nn as nn
import math

class MatchAttnMoEModel(nn.Module):
    def __init__(self, n_feat=13, d_model=128, n_heads=4, dropout=0.05):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model

        # Batch norm before and after attention
        self.bn_in = nn.BatchNorm1d(10)
        self.bn_out = nn.BatchNorm1d(10)
        self.dropout = nn.Dropout(dropout)

        # Multi‑head attention projections
        self.qs = nn.ModuleList([nn.Linear(n_feat, d_model) for _ in range(n_heads)])
        self.ks = nn.ModuleList([nn.Linear(n_feat, d_model) for _ in range(n_heads)])
        self.vs = nn.ModuleList([nn.Linear(n_feat, d_model) for _ in range(n_heads)])

        # Gating network for heads (Mixture of Experts weighting)
        self.gate = nn.Sequential(
            nn.Linear(n_feat, n_heads),  # 13 → 4  (inputs = feature count)
            nn.Softmax(dim=-1)
        )

        # Projection back to d_model per token
        self.proj = nn.Linear(d_model * n_heads, d_model)

        # Dense outputs
        self.fc1 = nn.Linear(10 * d_model, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 1)
        self.act = nn.ReLU()
        self.out = nn.Sigmoid()

        # xavier init
        self.apply(init_xavier)

    def forward(self, x):  # x: (B,10,13)
        B = x.size(0)
        x = self.bn_in(x)

        # attention heads
        heads_out = []
        for q, k, v in zip(self.qs, self.ks, self.vs):
            Q, K, V = q(x), k(x), v(x)
            scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(Q.size(-1))
            weights = torch.softmax(scores, dim=2)
            heads_out.append(torch.bmm(weights, V))  # (B,10,d_model)

        # calculate MoE gate weights per batch
        mean_in = x.mean(dim=1)               # (B, n_feat)
        gate_w = self.gate(mean_in)           # (B, n_heads)
        gate_w = gate_w.unsqueeze(1).unsqueeze(-1)  # (B,1,n_heads,1)

        # blend heads by their gate weights before concatenation
        heads_out = torch.stack(heads_out, dim=2)   # (B,10,n_heads,d_model)
        heads_out = heads_out * gate_w
        heads_out = heads_out.view(B, 10, -1)       # (B,10,n_heads*d_model)

        # projection + norm + dropout
        proj = self.proj(heads_out)
        proj = self.bn_out(proj)
        proj = self.dropout(proj)

        # flatten and dense
        flat = proj.flatten(start_dim=1)            # (B, 10*d_model)
        h = self.act(self.bn_fc1(self.fc1(flat)))
        y = self.out(self.fc2(h))                   # (B,1)
        return y
    
def init_xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,} | Trainable: {trainable:,}")

if __name__ == "__main__":
    model = MatchAttnMoEModel()
    count_params(model)