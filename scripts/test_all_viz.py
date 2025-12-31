"""
scripts/test_all_viz.py
=======================

Smoke‑test for all visualizations and NN inference.
Generates one example image per visualization and saves
them to notes/img/.
"""

import os, shutil, matplotlib.pyplot as plt
from datetime import datetime
from crawler.match_base import MatchBase
from core.entities import Player, Team, Match
from viz.player_profile import PlayerProfile
from viz.gold_map import GoldMap
from viz.spider_stats import SpiderStats
from viz.gold_contribution import GoldContribution
from viz.nn_infer import predict_match_outcome


def clean_output_dir(path="notes/img"):
    """Remove previous visualization artifacts for deterministic outputs."""
    if not os.path.isdir(path):
        return
    for entry in os.listdir(path):
        target = os.path.join(path, entry)
        if os.path.isfile(target) or os.path.islink(target):
            os.remove(target)
        elif os.path.isdir(target):
            shutil.rmtree(target)
    print(f"🧹  Cleared {path}/ before regenerating assets.")


def save_fig(fig, name):
    """Save figure under notes/img/ with timestamp."""
    out_dir = "notes/img"
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"{name}_{ts}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"📸  Saved → {path}")


def run_all(live_path="data/match_base/live.db",
            model_path="weights/moeT_003.pt"):

    clean_output_dir()
    mb = MatchBase(live_path=live_path)
    conn = mb.db.conn

    # Ten players for a test match
    puuids = [r[0] for r in conn.execute("SELECT puuid FROM players LIMIT 10")]
    blue, red = Team([Player(p) for p in puuids[:5]]), Team([Player(p) for p in puuids[5:10]])
    match = Match(blue, red)

    tests = [
        ("PlayerProfile",    PlayerProfile(mb),  blue.top),
        ("GoldMap",          GoldMap(mb),        match),
        ("SpiderStats",      SpiderStats(mb),    match),
        ("GoldContribution", GoldContribution(mb), match)
    ]

    # Run each visualization
    for name, viz, arg in tests:
        try:
            print(f"\n▶️  Running {name} …")
            fig = viz.build_figure(arg)
            if fig:
                save_fig(fig, name)
                print(f"✅  {name} OK")
            else:
                print(f"⚠️  {name} returned None")
        except Exception as e:
            print(f"❌  {name} failed → {e}")

    # NN inference check
    try:
        print("\n▶️  Testing NN inference …")
        pred = predict_match_outcome(mb, match, model_path)
        print(f"✅  Model prediction: {pred:.4f}")
    except Exception as e:
        print(f"❌  NN inference failed → {e}")


if __name__ == "__main__":
    run_all()