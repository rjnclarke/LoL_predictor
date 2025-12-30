"""
viz/gold_map.py
===============

GoldMap – visualizes team gold distribution and player contributions.

Input
-----
A `Match` object containing two `Team` instances.
Each team has 5 Players in canonical order: TOP → UTILITY.

Output
------
Side‑by‑side heat‑maps:
    x‑axis – players (by role)
    y‑axis – matches (1–10)
    color  – player_gold / team_gold ratios

Interpretation:
    • balanced teams → uniform vertical bands
    • carry‑heavy → bright, uneven columns
"""

from .base_viz import ABCViz
from core.entities import Match, Team
import pandas as pd, numpy as np, json
import matplotlib.pyplot as plt, seaborn as sns


class GoldMap(ABCViz):
    """Heat‑map comparing gold distribution for both teams."""

    # ------------------------------------------------------------------ #
    def fetch_data(self, match: Match):
        """
        Query last 10 matches for every player in the two teams and compute
        their gold ratio toward team & total gold.

        Returns
        -------
        dict → {'blue': DataFrame, 'red': DataFrame}
        """
        def _collect(team):
            recs = []
            for player in team:
                cur = self.conn.execute("""
                    SELECT stats_json, timestamp
                    FROM player_match_stats
                    WHERE puuid=?
                    ORDER BY timestamp DESC
                    LIMIT 10
                """, (player.puuid,))
                rows = cur.fetchall()
                for idx, (blob, ts) in enumerate(rows):
                    try:
                        d = json.loads(blob)
                    except Exception:
                        continue
                    recs.append({
                        "puuid": player.puuid,
                        "timestamp": ts,
                        "match_idx": idx + 1,
                        "gold": d.get("gold", 0)
                    })
            return pd.DataFrame(recs)

        blue_df = _collect(match.blue)
        red_df  = _collect(match.red)
        all_df  = pd.concat([blue_df, red_df], ignore_index=True)

        # compute ratios per match timestamp
        results = []
        for ts, group in all_df.groupby("timestamp"):
            gold_blue = group[group["puuid"].isin(match.blue.puuids)]["gold"].sum()
            gold_red  = group[group["puuid"].isin(match.red.puuids)]["gold"].sum()
            total = gold_blue + gold_red
            for _, row in group.iterrows():
                side = "blue" if row["puuid"] in match.blue.puuids else "red"
                team_gold = gold_blue if side == "blue" else gold_red
                results.append({
                    "puuid": row["puuid"],
                    "side": side,
                    "match_idx": row["match_idx"],
                    "player_share": row["gold"] / team_gold if team_gold else 0,
                    "team_share": team_gold / total if total else 0,
                })
        df = pd.DataFrame(results)
        return {
            "blue": df[df.side == "blue"],
            "red":  df[df.side == "red"]
        }

    # ------------------------------------------------------------------ #
    def build_figure(self, match: Match):
        """Render side‑by‑side heat‑maps comparing gold balance."""
        data = self.fetch_data(match)
        fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

        for ax, side, title in zip(axes, ["blue", "red"], ["Blue Team", "Red Team"]):
            df = data[side]
            if df.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                continue

            # create pivot: match_idx × player (role order)
            team_obj = match.team(side)
            pivot = df.pivot_table(
                index="match_idx",
                columns="puuid",
                values="player_share",
                aggfunc="mean"
            )[team_obj.puuids]  # ensure consistent order

            sns.heatmap(
                pivot,
                cmap="YlOrBr",
                vmin=0, vmax=0.4,
                cbar=True, ax=ax
            )
            ax.set_title(title)
            ax.set_xlabel("Players (by role)")
            ax.set_ylabel("Match index")
            ax.tick_params(axis="x", labelrotation=45)

        plt.suptitle("Gold Share Heat Map (Player / Team)", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        return fig


# ---------------------------------------------------------------------- #
# Stand‑alone test
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    from crawler.match_base import MatchBase
    from core.entities import Player, Team, Match
    import matplotlib.pyplot as plt

    mb = MatchBase(live_path="data/match_base_1/live.db")

    # Simple example: pick 10 players (5 per team)
    puuids = [r[0] for r in mb.db.conn.execute("SELECT puuid FROM players LIMIT 10")]
    blue_team = Team([Player(p) for p in puuids[:5]])
    red_team  = Team([Player(p) for p in puuids[5:10]])
    match = Match(blue_team, red_team)

    viz = GoldMap(mb)
    fig = viz.build_figure(match)
    plt.show()