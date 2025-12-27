import torch
import torch.nn as nn

class MatchMLPBaseline(nn.Module):
    def __init__(self, in_dim=10*13, hidden=[256,128,64], dropout=0.1):
        super().__init__()
        layers=[]
        last=in_dim
        for h in hidden:
            layers += [
                nn.Linear(last,h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            last=h
        layers += [nn.Linear(last,1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

        # xavier init
        self.net.apply(lambda m: nn.init.xavier_uniform_(m.weight)
                       if isinstance(m, nn.Linear) else None)

    def forward(self, x):             # x:(B,10,13)
        x = x.view(x.size(0), -1)     # flatten
        return self.net(x)
    
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,} | Trainable: {trainable:,}")

if __name__ == "__main__":
    model = MatchMLPBaseline()
    count_params(model)