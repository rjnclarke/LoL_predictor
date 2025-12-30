"""
viz/player_profile.py
=====================
PlayerProfile visualization – now expects a Player object.
"""

from .base_viz import ABCViz
from core.entities import Player
import json, pandas as pd, numpy as np, matplotlib.pyplot as plt

class PlayerProfile(ABCViz):
    """Displays a single player's static and dynamic statistics."""

    def fetch_data(self, player: Player):
        self.log(f"Fetching data for {player.puuid[:8]}...")
        cur = self.conn.execute(
            "SELECT tier FROM players WHERE puuid=?", (player.puuid,)
        )
        row = cur.fetchone()
        static = {"tier": row[0] if row else None}
        print(f"PLAYER {player.puuid[:8]}: TIER = {static['tier']}")

        cur = self.conn.execute("""
            SELECT stats_json, timestamp
            FROM player_match_stats
            WHERE puuid=? ORDER BY timestamp DESC LIMIT 10
        """, (player.puuid,))
        recs = [
            {**json.loads(js), "timestamp": ts, "match_num": i+1}
            for i, (js, ts) in enumerate(cur.fetchall())
        ]
        df = pd.DataFrame(recs)
        if df.empty:
            return {"static": static, "matches": None, "averages": None}

        avgs = df[["kills", "deaths", "assists", "gold", "damage", "vision"]].mean().to_dict()
        avgs["cs_per_min"] = (df["kills"] + df["assists"]) / 10
        return {"static": static, "matches": df, "averages": avgs}

    def build_figure(self, player: Player):
        data = self.fetch_data(player)
        df = data["matches"]
        if df is None or df.empty:
            print("❌ No data to plot.")
            return None

        fig = plt.figure(figsize=(10,5))
        plt.suptitle(f"Player Profile: {player.puuid[:8]}", fontsize=14, fontweight="bold")

        ax1 = fig.add_subplot(1,2,1)
        ax1.plot(df["match_num"], df["kills"], label="Kills", color="tab:red")
        ax1.plot(df["match_num"], df["deaths"], label="Deaths", color="tab:gray")
        ax1.plot(df["match_num"], df["assists"], label="Assists", color="tab:blue")
        ax1.set_xlabel("Match (Old → Recent)")
        ax1.set_ylabel("Count per match")
        ax1.legend()
        ax1.set_title("K/D/A Trend")

        stats = data["averages"]
        cats = ["gold", "cs_per_min", "damage", "vision"]
        vals = [stats.get(c,0) for c in cats] + [stats.get(cats[0],0)]
        angs = np.linspace(0, 2*np.pi, len(cats), endpoint=False).tolist()
        angs += angs[:1]
        ax2 = fig.add_subplot(1,2,2, polar=True)
        ax2.plot(angs, vals, "o-", linewidth=2)
        ax2.fill(angs, vals, alpha=0.25)
        ax2.set_thetagrids(np.degrees(angs[:-1]), cats)
        ax2.set_title("Average Performance")

        fig.tight_layout(rect=[0,0,1,0.93])
        return fig

# simple manual test
if __name__ == "__main__":
    from crawler.match_base import MatchBase
    import matplotlib.pyplot as plt
    mb = MatchBase(live_path="data/match_base_1/live.db")
    puuid = mb.db.conn.execute("SELECT puuid FROM players LIMIT 1").fetchone()[0]
    player = Player(puuid)
    viz = PlayerProfile(mb)
    fig = viz.build_figure(player)
    plt.show()