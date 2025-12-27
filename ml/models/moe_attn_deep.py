# ml/models/moe_transformer_deep.py
import torch
import torch.nn as nn
import math

# -------------------------------------------------------------
# single attention + FFN sub‑block
# -------------------------------------------------------------
class AttnBlock(nn.Module):
    def __init__(self, n_feat, d_model=128, n_heads=4, dropout=0.05):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model

        self.qs = nn.ModuleList([nn.Linear(n_feat, d_model) for _ in range(n_heads)])
        self.ks = nn.ModuleList([nn.Linear(n_feat, d_model) for _ in range(n_heads)])
        self.vs = nn.ModuleList([nn.Linear(n_feat, d_model) for _ in range(n_heads)])

        # gating on raw input features (13)
        self.gate = nn.Sequential(
            nn.Linear(n_feat, n_heads),
            nn.Softmax(dim=-1)
        )

        self.proj = nn.Linear(d_model * n_heads, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):                            # x:(B,10,n_feat)
        B = x.size(0)

        # --- Multi‑head attention with MoE gating
        heads_out = []
        for q, k, v in zip(self.qs, self.ks, self.vs):
            Q, K, V = q(x), k(x), v(x)
            attn = torch.bmm(Q, K.transpose(1,2)) / math.sqrt(Q.size(-1))
            attn = torch.softmax(attn, dim=2)
            heads_out.append(torch.bmm(attn, V))

        gate_w = self.gate(x.mean(dim=1)).unsqueeze(1).unsqueeze(-1)  # (B,1,n_heads,1)
        heads_out = torch.stack(heads_out, dim=2) * gate_w            # (B,10,n_heads,d_model)
        heads_out = heads_out.view(B, 10, -1)                         # (B,10,n_heads*d_model)

        out = self.proj(heads_out)                                    # (B,10,d_model)
        x = self.norm1(out + x)                                       # residual
        f = self.ffn(x)
        x = self.norm2(x + self.dropout(f))
        return x


# -------------------------------------------------------------
# 4‑layer deep attention model
# -------------------------------------------------------------
class MatchAttnMoEDeep(nn.Module):
    def __init__(self, n_feat=13, d_model=64, n_heads=16, n_layers=16, dropout=0.03):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(10)
        self.input_proj = nn.Linear(n_feat, d_model)
        self.layers = nn.ModuleList([
            AttnBlock(d_model, d_model=d_model, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.output_bn = nn.BatchNorm1d(10)
        self.output_drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(10 * d_model, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 1)
        self.act = nn.ReLU()
        self.out = nn.Sigmoid()

        self.apply(self._init_xavier)

    def _init_xavier(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):                           # (B,10,13)
        x = self.input_bn(x)
        x = self.input_proj(x)                      # (B,10,d_model)

        for blk in self.layers:
            x = blk(x)                              # stacked blocks

        x = self.output_bn(x)
        x = self.output_drop(x)
        flat = x.flatten(start_dim=1)
        h = self.act(self.bn1(self.fc1(flat)))
        y = self.out(self.fc2(h))
        return y
    
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,} | Trainable: {trainable:,}")

if __name__ == "__main__":
    model = MatchAttnMoEDeep()
    count_params(model)