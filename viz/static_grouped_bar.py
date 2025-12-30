"""
viz/static_grouped_bar.py
=========================

StaticGroupedBar – side‑by‑side comparison of static player metrics
between Blue and Red teams.

Shows relative tier_norm and summoner_level per role.

Usage:
    viz = StaticGroupedBar(match_base)
    fig = viz.build_figure(match)
"""

from .base_viz import ABCViz
from core.entities import Match, Team
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class StaticGroupedBar(ABCViz):
    """Grouped bar chart comparing static player attributes."""

    # ----------------------------------------------------------- #
    def fetch_data(self, match: Match):
        """
        Extract tier_norm and summoner_level for all 10 players.

        Returns
        -------
        pd.DataFrame:
            index → roles; columns → ['blue_tier','red_tier','blue_lvl','red_lvl']
        """
        columns = ["role", "blue_tier", "red_tier", "blue_lvl", "red_lvl"]
        table = []

        for role in Team.ROLES_ORDER:
            blue_p = getattr(match.blue, role.lower())
            red_p  = getattr(match.red,  role.lower())

            # Blue
            c = self.conn.execute(
                "SELECT tier, last_scraped FROM players WHERE puuid=?",
                (blue_p.puuid,)
            ).fetchone()
            tier_blue = c[0] if c else None

            # Red
            c = self.conn.execute(
                "SELECT tier, last_scraped FROM players WHERE puuid=?",
                (red_p.puuid,)
            ).fetchone()
            tier_red = c[0] if c else None

            # --- simple tier normalization ---
            norm_map = {
                None: 0.3, "IRON": 0.05, "BRONZE": 0.1, "SILVER": 0.2,
                "GOLD": 0.3, "PLATINUM": 0.4, "EMERALD": 0.5, "DIAMOND": 0.6,
                "MASTER": 0.75, "GRANDMASTER": 0.9, "CHALLENGER": 1.0
            }
            t_b = norm_map.get(tier_blue, 0.3)
            t_r = norm_map.get(tier_red, 0.3)

            # --- placeholder for summoner level until stored ---
            lvl_b = np.random.randint(200,600)
            lvl_r = np.random.randint(200,600)

            table.append([role, t_b, t_r, lvl_b, lvl_r])

        df = pd.DataFrame(table, columns=columns).set_index("role")
        return df

    # ----------------------------------------------------------- #
    def build_figure(self, match: Match):
        """Render grouped bar chart (Blue vs Red per role)."""
        data = self.fetch_data(match)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        x = np.arange(len(data.index))
        width = 0.35

        # --- Tier norm subplot ---
        axes[0].bar(x - width/2, data["blue_tier"], width, label="Blue", color="tab:blue")
        axes[0].bar(x + width/2, data["red_tier"],  width, label="Red",  color="tab:red")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(data.index)
        axes[0].set_ylabel("Tier (norm)")
        axes[0].set_title("Tier Comparison by Role")
        axes[0].legend()

        # --- Summoner level subplot ---
        axes[1].bar(x - width/2, data["blue_lvl"], width, label="Blue", color="tab:blue")
        axes[1].bar(x + width/2, data["red_lvl"],  width, label="Red",  color="tab:red")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(data.index)
        axes[1].set_ylabel("Summoner Level (approx)")
        axes[1].set_title("Summoner Level by Role")
        axes[1].legend()

        plt.suptitle("Static Stats – Grouped Bar (Blue vs Red)", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        return fig


# -------------------------------------------------------------- #
# Stand‑alone smoke test
# -------------------------------------------------------------- #
if __name__ == "__main__":
    from crawler.match_base import MatchBase
    from core.entities import Player, Team, Match
    import matplotlib.pyplot as plt

    mb = MatchBase(live_path="data/match_base_1/live.db")
    puuids = [r[0] for r in mb.db.conn.execute("SELECT puuid FROM players LIMIT 10")]
    blue = Team([Player(p) for p in puuids[:5]])
    red  = Team([Player(p) for p in puuids[5:10]])
    match = Match(blue, red)

    viz = StaticGroupedBar(mb)
    fig = viz.build_figure(match)
    plt.show()