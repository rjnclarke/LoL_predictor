"""
viz/gold_contribution.py
========================

GoldContribution – each player's relative gold contribution
within their own team, normalized so each team’s 5 bars sum to 1 .0.
"""

from .base_viz import ABCViz
from core.entities import Match, Team
import json, pandas as pd, numpy as np, matplotlib.pyplot as plt

ROLE_COLORS = {
    "TOP": "#5470C6",
    "JUNGLE": "#91CC75",
    "MIDDLE": "#EE6666",
    "BOTTOM": "#FAC858",
    "UTILITY": "#73C0DE",
}

class GoldContribution(ABCViz):
    """Team‑normalized gold‑share bar chart."""

    def fetch_data(self, match: Match):
        """Compute mean player_gold/team_gold for last 10 matches, normalize per team."""
        def _team_avg(team):
            vals = []
            for p in team:
                rows = self.conn.execute("""
                    SELECT stats_json FROM player_match_stats
                    WHERE puuid=? ORDER BY timestamp DESC LIMIT 10
                """,(p.puuid,)).fetchall()
                if not rows:
                    vals.append(0); continue
                golds=[json.loads(j[0]).get("gold",0) for j in rows]
                vals.append(np.mean(golds))
            tot=sum(vals)
            return [v/tot if tot else 0 for v in vals]

        blue=_team_avg(match.blue)
        red =_team_avg(match.red)
        return pd.DataFrame({"blue":blue,"red":red},index=Team.ROLES_ORDER)

    def build_figure(self, match: Match):
        data = self.fetch_data(match)
        fig, ax = plt.subplots(figsize=(6, 6))

        bars_x = {"blue": -0.2, "red": 0.2}
        for side, x in bars_x.items():
            bottom = 0
            shares = data[side]
            for role, share in zip(Team.ROLES_ORDER, shares):
                ax.bar(
                    x,
                    share,
                    width=0.35,
                    bottom=bottom,
                    color=ROLE_COLORS.get(role, "#999"),
                    edgecolor="white",
                    linewidth=0.5,
                    label=role if side == "blue" else None,
                )
                ax.text(
                    x,
                    bottom + share / 2,
                    f"{share*100:,.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#1f1f1f",
                )
                bottom += share

        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(0, 1)
        ax.set_xticks([bars_x["blue"], bars_x["red"]], ["Blue", "Red"])
        ax.set_ylabel("Share of team gold (sums to 100%)")
        ax.set_title("Gold Contribution – Stacked by Role")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax.legend(title="Role", loc="upper right")
        fig.tight_layout()
        return fig


# Stand‑alone test
if __name__=="__main__":
    from crawler.match_base import MatchBase
    from core.entities import Player,Team,Match
    import matplotlib.pyplot as plt
    mb=MatchBase(live_path="data/match_base_1/live.db")
    puuids=[r[0] for r in mb.db.conn.execute("SELECT puuid FROM players LIMIT 10")]
    blue,red=Team([Player(p) for p in puuids[:5]]),Team([Player(p) for p in puuids[5:10]])
    match=Match(blue,red)
    viz=GoldContribution(mb)
    fig=viz.build_figure(match); plt.show()