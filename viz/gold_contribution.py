"""
viz/gold_contribution.py
========================

GoldContribution – each player's relative gold contribution
within their own team, normalized so each team’s 5 bars sum to 1 .0.
"""

from .base_viz import ABCViz
from core.entities import Match, Team
import json, pandas as pd, numpy as np, matplotlib.pyplot as plt

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
        data=self.fetch_data(match)
        x=np.arange(len(Team.ROLES_ORDER)); width=0.35
        fig,ax=plt.subplots(figsize=(8,5))
        ax.bar(x-width/2,data.blue,width,label="Blue",color="tab:blue")
        ax.bar(x+width/2,data.red ,width,label="Red" ,color="tab:red")
        ax.set_xticks(x); ax.set_xticklabels(Team.ROLES_ORDER)
        ax.set_ylabel("Normalized Gold Share")
        ax.set_ylim(0,1)
        ax.set_title("Gold Contribution per Role (Normalized per Team)")
        ax.legend(); fig.tight_layout()
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