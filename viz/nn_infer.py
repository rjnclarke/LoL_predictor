"""Model inference helpers + visualization."""

import json, torch, pandas as pd, numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from core.entities import Team, Match
from ml.models.moe_transformer import MatchAttnMoEModel


# ----------------------------------------------------------- #
# Safe number helper
# ----------------------------------------------------------- #
def safe_num(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return 0.0
        return float(x)
    except Exception:
        return 0.0


# ----------------------------------------------------------- #
# Feature builders
# ----------------------------------------------------------- #
def build_player_vector(conn, puuid):
    """
    Construct the 13‑feature numeric vector for one player
    directly from the live DB.
    """
    # ---- Static ----
    srow = conn.execute(
        "SELECT tier FROM players WHERE puuid=?", (puuid,)
    ).fetchone()
    tier = srow[0] if srow else None
    tier_norm_map = {
        None: 0.3, "IRON": 0.05, "BRONZE": 0.1, "SILVER": 0.2,
        "GOLD": 0.3, "PLATINUM": 0.4, "EMERALD": 0.5,
        "DIAMOND": 0.6, "MASTER": 0.75, "GRANDMASTER": 0.9, "CHALLENGER": 1.0
    }
    tier_norm = safe_num(tier_norm_map.get(tier, 0.3))

    # ---- Dynamic averages ----
    cur = conn.execute("""
        SELECT stats_json FROM player_match_stats
        WHERE puuid=? ORDER BY timestamp DESC LIMIT 10
    """, (puuid,))
    rows = cur.fetchall()
    if not rows:
        return [0.0] * 13

    stats = [json.loads(j[0]) for j in rows]
    df = pd.DataFrame(stats)

    kills   = safe_num(df["kills"].mean())
    deaths  = safe_num(df["deaths"].mean())
    assists = safe_num(df["assists"].mean())
    gold_pm = safe_num(df["gold"].mean())
    cs_pm   = safe_num(((df["kills"] + df["assists"]) / 10).mean())
    vision  = safe_num(df["vision"].mean())
    damage  = safe_num(df["damage"].mean())
    win_r   = 0.5  # placeholder until stored

    feats = [
        tier_norm, 0.0, 0.0, 0.0, 0.0,
        kills, deaths, assists,
        gold_pm, cs_pm,
        vision, damage, win_r,
    ]
    return feats


def build_team_tensor(match_base, team: Team):
    """Return [5,13] tensor for the given team."""
    conn = match_base.db.conn
    rows = [build_player_vector(conn, p.puuid) for p in team]
    return torch.tensor(rows, dtype=torch.float32)


def build_match_tensor(match_base, match: Match):
    """Concatenate the two team tensors → [10,13] match tensor."""
    blue_tensor = build_team_tensor(match_base, match.blue)
    red_tensor  = build_team_tensor(match_base, match.red)
    return torch.cat([blue_tensor, red_tensor], dim=0)


# ----------------------------------------------------------- #
# Model loading + inference
# ----------------------------------------------------------- #
def load_model(weights_path="weights/moeT_003.pt", device=None):
    """
    Load MatchAttnMoEModel weights (supports state_dict or full model).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = MatchAttnMoEModel()
    checkpoint = torch.load(weights_path, map_location=device)

    # handle both full model and pure state_dict
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        model = checkpoint

    model.to(device).eval()
    return model, device


@torch.no_grad()
def predict_match_outcome(match_base, match: Match,
                          model_path="weights/moeT_003.pt"):
    """Return raw regression output plus clipped blue/red success scores."""
    X = build_match_tensor(match_base, match).unsqueeze(0)
    model, device = load_model(model_path)
    X = X.to(device)
    y_pred = model(X)
    raw = float(y_pred.squeeze().cpu().item())
    blue_success = float(np.clip(raw, 0.0, 1.0))
    return {
        "raw": raw,
        "blue_success": blue_success,
        "red_success": 1.0 - blue_success,
    }


def build_speedometer(blue_success: float):
    """Render a horizontal split bar (blue vs red share)."""
    blue_success = float(np.clip(blue_success, 0.0, 1.0))
    red_success = 1.0 - blue_success

    fig, ax = plt.subplots(figsize=(6.5, 1.6))
    ax.barh(
        [0],
        [blue_success],
        color="#2563eb",
        edgecolor="white",
        linewidth=1.5,
    )
    ax.barh(
        [0],
        [red_success],
        left=blue_success,
        color="#dc2626",
        edgecolor="white",
        linewidth=1.5,
    )
    ax.axvline(0.5, color="#f8fafc", linestyle="--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], color="#e2e8f0")
    ax.set_xlabel("Blue share →", color="#e2e8f0", fontsize=10)
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")
    ax.text(
        blue_success / 2,
        0.05,
        f"Blue {blue_success*100:,.1f}%",
        ha="center",
        va="center",
        color="#e0f2fe",
        fontsize=11,
        fontweight="bold",
    )
    ax.text(
        blue_success + red_success / 2,
        0.05,
        f"Red {red_success*100:,.1f}%",
        ha="center",
        va="center",
        color="#fee2e2",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_title("Model inference (blue ↔ red success)", color="#f1f5f9", fontsize=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig


# ----------------------------------------------------------- #
# Smoke test
# ----------------------------------------------------------- #
if __name__ == "__main__":
    from crawler.match_base import MatchBase
    from core.entities import Player, Team, Match

    mb = MatchBase(live_path="data/match_base/live.db")
    puuids = [r[0] for r in mb.db.conn.execute("SELECT puuid FROM players LIMIT 10")]
    blue = Team([Player(p) for p in puuids[:5]])
    red  = Team([Player(p) for p in puuids[5:10]])
    match = Match(blue, red)

    result = predict_match_outcome(mb, match)
    print(
        f"Raw score: {result['raw']:.4f} | Blue success {result['blue_success']*100:,.2f}%"
    )
    gauge = build_speedometer(result["blue_success"])
    plt.show()