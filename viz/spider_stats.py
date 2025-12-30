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
                avgs = df[["gold","kills","deaths","assists","damage","vision"]].mean()
                recs.append(avgs)
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

        metrics = data.index.tolist()
        n = len(metrics)
        angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist()
        angles += angles[:1]  # close circle

        # compute values
        values_blue = data.loc[:, "blue"].tolist()
        values_red  = data.loc[:, "red"].tolist()
        values_blue += values_blue[:1]
        values_red  += values_red[:1]

        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles, values_blue, color="tab:blue", linewidth=2, label="Blue Team")
        ax.fill(angles, values_blue, color="tab:blue", alpha=0.25)
        ax.plot(angles, values_red,  color="tab:red", linewidth=2, label="Red Team")
        ax.fill(angles, values_red,  color="tab:red", alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
        ax.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))
        ax.set_title("Team Performance Comparison", fontsize=13, pad=20)
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