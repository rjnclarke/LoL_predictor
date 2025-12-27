# ml/run.py
import torch, os, json, shutil
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from models.mlp import MatchMLPBaseline
from models.moe_transformer import MatchAttnMoEModel
from models.moe_attn_deep import MatchAttnMoEDeep


# -------------------------------------------------------------
# Utility helpers
# -------------------------------------------------------------
def load_split(path):
    X, y = torch.load(path, map_location="cpu")
    return TensorDataset(X, y)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_curves(hist, out_path):
    plt.figure()
    plt.plot(hist["train_loss"], label="train")
    plt.plot(hist["val_loss"], label="val")
    plt.xlabel("epoch"); plt.ylabel("MSE")
    plt.title("Loss Curves")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_path); plt.close()


# -------------------------------------------------------------
# Training / evaluation
# -------------------------------------------------------------
def evaluate(model, loader, device, criterion):
    model.eval(); total = 0.0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            total += criterion(model(X), y).item() * X.size(0)
    return total / len(loader.dataset)


def train_one(model, train_loader, val_loader, device, lr, wd,
              max_epochs=100, patience=25):

    model.to(device)
    criterion = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_val = float("inf")
    best_state = None
    hist = {"train_loss": [], "val_loss": []}
    wait = 0

    for e in range(max_epochs):
        model.train()
        tloss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optim.zero_grad()
            l = criterion(model(X), y)
            l.backward()
            optim.step()
            tloss += l.item() * X.size(0)
        tloss /= len(train_loader.dataset)
        vloss = evaluate(model, val_loader, device, criterion)
        hist["train_loss"].append(tloss)
        hist["val_loss"].append(vloss)
        print(f"Epoch {e+1:03d}: train={tloss:.5f}  val={vloss:.5f}")

        if vloss < best_val:
            best_val = vloss
            best_state = model.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("⏸ Early stop.\n")
                break

    model.load_state_dict(best_state)
    return best_val, hist, model


# -------------------------------------------------------------
# Run manager / hyper‑parameter search
# -------------------------------------------------------------
def run_experiment(model_cls, run_prefix, hypers):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_data = load_split("data/train_cpu.pt")
    dev_data   = load_split("data/dev_cpu.pt")

    best_overall = {"best_val": float("inf")}
    run_id = 0
    results = []

    for bs in hypers["batch_size"]:
        for lr in hypers["lr"]:
            for wd in hypers["weight_decay"]:
                run_id += 1
                run_name = f"{run_prefix}_{run_id:03d}"
                run_dir  = os.path.join("runs", run_name)
                os.makedirs(run_dir, exist_ok=True)

                print("\n---------------------------------------------")
                print(f"🚀 {run_name}")
                print(f"   batch_size={bs}, lr={lr}, weight_decay={wd}")

                model = model_cls()
                print(f"   parameters={count_params(model):,}")

                train_loader = DataLoader(train_data, bs, shuffle=True)
                val_loader   = DataLoader(dev_data,  bs, shuffle=False)

                best_val, hist, trained = train_one(
                    model, train_loader, val_loader, device,
                    lr=lr, wd=wd, max_epochs=60, patience=6
                )

                pd.DataFrame(hist).to_csv(f"{run_dir}/history.csv", index=False)
                plot_curves(hist, f"{run_dir}/curves.png")
                meta = {
                    "run": run_name,
                    "batch_size": bs,
                    "lr": lr,
                    "weight_decay": wd,
                    "best_val": best_val,
                    "epochs": len(hist["train_loss"]),
                }
                results.append(meta)

                if best_val < best_overall["best_val"]:
                    best_overall = meta.copy()
                    best_overall["state_dict"] = trained.state_dict()
                    print(f"🌟 New best run: {run_name} (val={best_val:.5f})")

    # save only best overall
    os.makedirs("weights", exist_ok=True)
    best_name = best_overall["run"]
    best_w_path = f"weights/{best_name}.pt"
    torch.save(best_overall["state_dict"], best_w_path)

    meta_no_state = {k: v for k, v in best_overall.items() if k != "state_dict"}
    json.dump(meta_no_state, open(f"runs/{best_name}_best.json","w"), indent=2)
    pd.DataFrame(results).to_csv("runs/summary.csv", index=False)

    print("\n==============================")
    print("✅ Finished grid search")
    print(f"Best Run    : {best_name}")
    print(f"Val loss    : {best_overall['best_val']:.6f}")
    print(f"batch_size  = {best_overall['batch_size']}")
    print(f"lr          = {best_overall['lr']}")
    print(f"weight_decay= {best_overall['weight_decay']}")
    print(f"Weights saved -> {best_w_path}")
    print("==============================\n")


# -------------------------------------------------------------
# Wipe runs and weights (confirmation prompt)
# -------------------------------------------------------------
def wipe_runs():
    confirm = input(
        "🔥 This will DELETE all runs/ and weights/ permanently.\n"
        "Type 'yes' to confirm: "
    ).strip().lower()
    if confirm != "yes":
        print("Abort.")
        return
    for folder in ["runs", "weights"]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"Deleted: {folder}/")
    print("✅ All run folders and weights wiped.")


# -------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------
if __name__ == "__main__":
    hypers = {
        "batch_size": [16],
        "lr": [0.0004],
        "weight_decay": [0.0003]
    }

    # Uncomment below to clear previous results
    # wipe_runs()

    # choose model to train
    #run_experiment(MatchAttnMoEModel, "moeT", hypers)
    #run_experiment(MatchMLPBaseline, "mlp", hypers)
    run_experiment(MatchAttnMoEDeep, "moeDeepT.2", hypers)