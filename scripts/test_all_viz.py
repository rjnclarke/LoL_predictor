"""
scripts/test_all_viz.py
=======================

Smoke test for all visualization modules + NN inference.

It:
  • opens data/match_base/live.db
  • picks 10 players (5v5)
  • runs each viz' build_figure()
  • calls predict_match_outcome()
"""

import matplotlib.pyplot as plt
from crawler.match_base import MatchBase
from core.entities import Player, Team, Match
from viz.player_profile import PlayerProfile
from viz.gold_map import GoldMap
from viz.spider_stats import SpiderStats
from viz.static_grouped_bar import StaticGroupedBar
from viz.gold_contribution import GoldContribution
from viz.nn_infer import predict_match_outcome


def run_all(live_path="data/match_base/live.db",
            model_path="weights/moeT_003.pt"):

    mb = MatchBase(live_path=live_path)
    conn = mb.db.conn

    # ten players for a test match
    puuids = [r[0] for r in conn.execute("SELECT puuid FROM players LIMIT 10")]
    blue, red = Team([Player(p) for p in puuids[:5]]), Team([Player(p) for p in puuids[5:10]])
    match = Match(blue, red)

    tests = [
        ("PlayerProfile",    PlayerProfile(mb),  blue.top),  # single player
        ("GoldMap",          GoldMap(mb),        match),
        ("SpiderStats",      SpiderStats(mb),    match),
        ("StaticGroupedBar", StaticGroupedBar(mb), match),
        ("GoldContribution", GoldContribution(mb), match)
    ]

    for name, viz, arg in tests:
        try:
            print(f"\n▶️  Running {name} …")
            fig = viz.build_figure(arg)
            if fig:
                plt.close(fig)
                print(f"✅  {name} OK")
            else:
                print(f"⚠️  {name} returned None")
        except Exception as e:
            print(f"❌  {name} failed → {e}")

    # NN inference check
    try:
        print("\n▶️  Testing NN inference")
        pred = predict_match_outcome(mb, match, model_path)
        print(f"✅ Model prediction: {pred:.4f}")
    except Exception as e:
        print(f"❌ NN inference failed → {e}")


if __name__ == "__main__":
    run_all()