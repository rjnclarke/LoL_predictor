"""
viz/player_profile.py
=====================
PlayerProfile visualization – now expects a Player object.
"""

from .base_viz import ABCViz
from core.entities import Player
import json, pandas as pd, numpy as np, matplotlib.pyplot as plt

RADAR_METRICS = [
    ("gold", "Gold Earned", 20000),
    ("damage", "Damage to Champs", 30000),
    ("cs_per_min", "CS / Min", 12),
    ("vision", "Vision Score", 60),
]

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
        avgs["cs_per_min"] = round(((df["kills"] + df["assists"]) / 10).mean(), 3)
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
        keys = [m[0] for m in RADAR_METRICS]
        raw_vals = [stats.get(k, 0) for k in keys]
        scaled_vals = []
        for raw, (_, _, maximum) in zip(raw_vals, RADAR_METRICS):
            scaled = raw / maximum if maximum else raw
            scaled_vals.append(max(0.0, min(scaled, 1.0)))
        scaled_vals += scaled_vals[:1]

        angs = np.linspace(0, 2*np.pi, len(RADAR_METRICS), endpoint=False).tolist()
        angs += angs[:1]

        ax2 = fig.add_subplot(1,2,2, polar=True)
        ax2.set_theta_offset(np.pi / 2)
        ax2.set_theta_direction(-1)
        ax2.set_ylim(0, 1)
        ax2.set_title("Average Performance (Normalized)")
        ax2.set_thetagrids(
            np.degrees(angs[:-1]),
            [label for _, label, _ in RADAR_METRICS]
        )
        ax2.set_rgrids([0.25, 0.5, 0.75, 1.0], angle=22.5, fontsize=8)
        ax2.plot(angs, scaled_vals, color="tab:green", linewidth=2)
        ax2.fill(angs, scaled_vals, alpha=0.2, color="tab:green")

        for angle, raw, (metric, _, _) in zip(angs[:-1], raw_vals, RADAR_METRICS):
            fmt = f"{raw:.1f}" if metric in {"cs_per_min", "vision"} else f"{raw:,.0f}"
            ax2.text(angle, 1.08, fmt, fontsize=8, ha="center", va="center", color="dimgray")

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