"""
viz/spider_stats.py
===================

SpiderStats – compares average per‑team performance.

Input:
    Match object with two Team instances (blue/red)
Metrics averaged over the last 10 matches of each player:
    gold, kills, deaths, assists, damage, vision score

Output:
    Overlayed radar (spider) chart – Blue vs Red team means.
"""

from .base_viz import ABCViz
from core.entities import Match, Team
import json, numpy as np, pandas as pd, matplotlib.pyplot as plt

METRICS = [
    ("gold", "Gold", 20000),
    ("damage", "Damage", 30000),
    ("vision", "Vision", 60),
    ("kda", "KDA", 6),
]

class SpiderStats(ABCViz):
    """Team‑comparison spider chart."""

    # ------------------------------------------------------------ #
    def fetch_data(self, match: Match):
        """Aggregate last 10 matches for each player, return team averages."""
        def _aggregate(team):
            recs = []
            for player in team:
                cur = self.conn.execute("""
                    SELECT stats_json FROM player_match_stats
                    WHERE puuid=? ORDER BY timestamp DESC LIMIT 10
                """, (player.puuid,))
                rows = cur.fetchall()
                if not rows:
                    continue
                stats = [json.loads(j[0]) for j in rows]
                df = pd.DataFrame(stats)
                if df.empty:
                    continue
                summary = pd.Series({
                    "gold": df["gold"].mean(),
                    "damage": df["damage"].mean(),
                    "vision": df["vision"].mean(),
                    "kda": (df["kills"].mean() + df["assists"].mean()) / max(1, df["deaths"].mean()),
                })
                recs.append(summary)
            if not recs:
                return pd.Series(dtype=float)
            team_avg = pd.concat(recs, axis=1).mean(axis=1)
            return team_avg

        blue_avg = _aggregate(match.blue)
        red_avg  = _aggregate(match.red)
        data = pd.DataFrame({"blue": blue_avg, "red": red_avg})
        return data

    # ------------------------------------------------------------ #
    def build_figure(self, match: Match):
        """Render a spider/radar plot comparing both teams."""
        data = self.fetch_data(match)
        if data.empty:
            print("❌ No data to plot for SpiderStats.")
            return None

        for metric, _, _ in METRICS:
            if metric not in data.index:
                data.loc[metric] = 0

        ordered = data.loc[[m[0] for m in METRICS]]
        normed = ordered.copy()
        for metric, _, max_val in METRICS:
            normed.loc[metric] = ordered.loc[metric] / max_val if max_val else ordered.loc[metric]
        normed = normed.clip(lower=0, upper=1)

        n = len(METRICS)
        angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist()
        angles += angles[:1]

        def _vals(side):
            values = normed.loc[:, side].tolist()
            values += values[:1]
            return values

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, polar=True)
        ax.set_theta_offset(np.pi / 4)
        ax.set_theta_direction(-1)

        vals_blue = _vals("blue")
        vals_red = _vals("red")
        ax.plot(angles, vals_blue, color="tab:blue", linewidth=2, label="Blue team")
        ax.fill(angles, vals_blue, color="tab:blue", alpha=0.2)
        ax.plot(angles, vals_red, color="tab:red", linewidth=2, label="Red team")
        ax.fill(angles, vals_red, color="tab:red", alpha=0.2)

        ax.set_thetagrids(np.degrees(angles[:-1]), [label for _, label, _ in METRICS], fontsize=11)
        ax.set_rgrids([0.25, 0.5, 0.75, 1.0], angle=-45, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title("Spider Stats – Normalized Team Profile", fontsize=13, pad=18)
        ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.1))
        fig.tight_layout()
        return fig


# ---------------------------------------------------------------- #
# Stand‑alone test
# ---------------------------------------------------------------- #
if __name__ == "__main__":
    from crawler.match_base import MatchBase
    from core.entities import Player, Team, Match
    import matplotlib.pyplot as plt

    mb = MatchBase(live_path="data/match_base_1/live.db")
    puuids = [r[0] for r in mb.db.conn.execute("SELECT puuid FROM players LIMIT 10")]
    blue_team = Team([Player(p) for p in puuids[:5]])
    red_team  = Team([Player(p) for p in puuids[5:10]])
    match = Match(blue_team, red_team)

    viz = SpiderStats(mb)
    fig = viz.build_figure(match)
    if fig: plt.show()