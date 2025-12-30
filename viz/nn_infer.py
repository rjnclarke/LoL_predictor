"""
viz/nn_infer.py
===============

Neural‑net inference utilities used within the visualization layer.

Builds the 13‑feature player/team/match tensors directly from the
current (live) database structure:
    players + player_match_stats tables

This keeps inference self‑contained and aligned with the trained model.
"""

import json, torch, pandas as pd, numpy as np
from core.entities import Team, Match

# ----------------------------------------------------------- #
# Safe conversion helpers
# ----------------------------------------------------------- #
def safe_num(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return 0.0
        return float(x)
    except Exception:
        return 0.0


# ----------------------------------------------------------- #
# Feature construction
# ----------------------------------------------------------- #
def build_player_vector(conn, puuid):
    """
    Construct a 13‑feature vector for one player directly from live DB.
    Pulls static info from `players` table and averages dynamic stats
    over the last 10 entries in `player_match_stats`.
    """
    # ---- Static fields ----
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

    # ---- Dynamic stats ----
    cur = conn.execute("""
        SELECT stats_json FROM player_match_stats
        WHERE puuid=? ORDER BY timestamp DESC LIMIT 10
    """, (puuid,))
    rows = cur.fetchall()
    if not rows:
        return [0.0]*13  # fallback if no records

    stats = [json.loads(j[0]) for j in rows]
    df = pd.DataFrame(stats)

    kills   = safe_num(df["kills"].mean())
    deaths  = safe_num(df["deaths"].mean())
    assists = safe_num(df["assists"].mean())
    gold_pm = safe_num(df["gold"].mean())
    cs_pm   = safe_num((df["kills"] + df["assists"]).mean()/10)
    vision  = safe_num(df["vision"].mean())
    damage  = safe_num(df["damage"].mean())
    win_r   = 0.5  # placeholder until win tracking added

    feats = [
        tier_norm,       # 1
        0.0,             # rank_strength (not in visual DB)
        0.0,             # summoner_level
        0.0,             # mastery_score
        0.0,             # challenge_points
        kills, deaths, assists,
        gold_pm, cs_pm,
        vision, damage, win_r
    ]
    return feats


def build_team_tensor(match_base, team: Team):
    """Return [5,13] tensor for one team based on current DB."""
    conn = match_base.db.conn
    rows = [build_player_vector(conn, p.puuid) for p in team]
    return torch.tensor(rows, dtype=torch.float32)


def build_match_tensor(match_base, match: Match):
    """Concatenate both teams → [10,13] tensor."""
    blue_tensor = build_team_tensor(match_base, match.blue)
    red_tensor  = build_team_tensor(match_base, match.red)
    return torch.cat([blue_tensor, red_tensor], dim=0)


# ----------------------------------------------------------- #
# Model loading & prediction
# ----------------------------------------------------------- #
def load_model(weights_path="weights/name_of_model.pt", device=None):
    """Load trained PyTorch model from .pt file."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(weights_path, map_location=device)
    model.eval()
    return model, device


@torch.no_grad()
def predict_match_outcome(match_base, match: Match,
                          model_path="weights/name_of_model.pt"):
    """
    Returns model prediction (probability or score depending on the model).
    """
    X = build_match_tensor(match_base, match).unsqueeze(0)  # [1,10,13]
    model, device = load_model(model_path)
    X = X.to(device)
    y_pred = model(X)
    return y_pred.squeeze().cpu().item()


# ----------------------------------------------------------- #
# Stand‑alone smoke test
# ----------------------------------------------------------- #
if __name__ == "__main__":
    from crawler.match_base import MatchBase
    from core.entities import Player, Team, Match

    mb = MatchBase(live_path="data/match_base_1/live.db")
    # simple test with first 10 players
    puuids = [r[0] for r in mb.db.conn.execute("SELECT puuid FROM players LIMIT 10")]
    blue_team = Team([Player(p) for p in puuids[:5]])
    red_team  = Team([Player(p) for p in puuids[5:10]])
    match = Match(blue_team, red_team)

    pred = predict_match_outcome(mb, match)
    print(f"Predicted outcome score: {pred:.4f}")