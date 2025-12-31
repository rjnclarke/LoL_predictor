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
Scatter field (≈50 points per team when data available):
    x-axis – player_gold ÷ team_gold (individual share)
    y-axis – team_gold ÷ total_gold (team dominance)
    color/marker – role + team indicator

Interpretation:
    • x close to 0.2 = balanced roster
    • x closer to 0.4+ = carry soaking gold
    • y close to 0.5 = even game, >0.6 = stomp
"""

from .base_viz import ABCViz
from core.entities import Match, Team
import json
import matplotlib.pyplot as plt

TEAM_COLORS = {"blue": "#4C78A8", "red": "#E45756"}
ROLE_MARKERS = {
    "TOP": "o",
    "JUNGLE": "^",
    "MIDDLE": "s",
    "BOTTOM": "D",
    "UTILITY": "P",
}


class GoldMap(ABCViz):
    """Scatter plot of player vs. team gold shares (last 10 matches × 5 players)."""

    def __init__(self, match_base):
        super().__init__(match_base)
        self._match_cache = {}
        self._stats_cache = {}

    # ------------------------------------------------------------------ #
    def fetch_data(self, match: Match, per_player: int = 10):
        """Return dict with blue/red point clouds or None if data missing."""
        payload = {"blue": [], "red": []}

        roster = {
            "blue": list(zip(Team.ROLES_ORDER, match.blue.players)),
            "red": list(zip(Team.ROLES_ORDER, match.red.players)),
        }

        for side, entries in roster.items():
            for role, player in entries:
                rows = self.conn.execute(
                    """
                    SELECT match_id, stats_json, timestamp
                    FROM player_match_stats
                    WHERE puuid=?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (player.puuid, per_player)
                ).fetchall()

                for order_idx, (match_id, stats_blob, _) in enumerate(rows, start=1):
                    stats = self._safe_json(stats_blob)
                    if not stats:
                        self.log(f"[GoldMap] Failed to parse stats for {player.puuid[:8]} match {match_id}")
                        continue
                    share = self._compute_shares(match_id, player.puuid, stats.get("gold", 0))
                    if not share:
                        self.log(f"[GoldMap] Missing share data for {player.puuid[:8]} match {match_id}")
                        continue
                    payload[side].append({
                        "role": role,
                        "role_idx": Team.ROLES_ORDER.index(role),
                        "match_rank": order_idx,
                        "player_share": share["player_share"],
                        "team_share": share["team_share"],
                    })

        return payload

    # ------------------------------------------------------------------ #
    def _compute_shares(self, match_id: str, puuid: str, player_gold: float):
        meta = self._match_meta(match_id)
        stats = self._match_stats(match_id)
        if not meta or not stats:
            return None

        if puuid in meta["blue"]:
            side, other = "blue", "red"
        elif puuid in meta["red"]:
            side, other = "red", "blue"
        else:
            return None

        if any(p not in stats for p in meta[side]) or any(p not in stats for p in meta[other]):
            return None

        team_gold = sum(stats[p] for p in meta[side])
        opponent_gold = sum(stats[p] for p in meta[other])
        total = team_gold + opponent_gold
        if team_gold <= 0 or total <= 0:
            return None

        return {
            "player_share": player_gold / team_gold,
            "team_share": team_gold / total,
        }

    def _match_meta(self, match_id: str):
        if match_id in self._match_cache:
            return self._match_cache[match_id]
        row = self.conn.execute(
            "SELECT puuids_json FROM matches WHERE match_id=?",
            (match_id,)
        ).fetchone()
        if not row or not row[0]:
            self.log(f"[GoldMap] Match {match_id} missing roster")
            return None
        try:
            players = json.loads(row[0])
        except Exception:
            self.log(f"[GoldMap] Match {match_id} roster JSON invalid")
            return None
        if len(players) != 10:
            self.log(f"[GoldMap] Match {match_id} roster length {len(players)} != 10")
            return None
        meta = {
            "blue": players[:5],
            "red": players[5:],
        }
        self._match_cache[match_id] = meta
        return meta

    def _match_stats(self, match_id: str):
        if match_id in self._stats_cache:
            return self._stats_cache[match_id]
        # Prefer denormalized gold stored on the matches table
        row = self.conn.execute(
            "SELECT player_gold_json FROM matches WHERE match_id=?",
            (match_id,)
        ).fetchone()
        if row and row[0]:
            payload = self._safe_json(row[0])
            if isinstance(payload, dict) and len(payload) >= 10:
                self._stats_cache[match_id] = payload
                return payload

        rows = self.conn.execute(
            "SELECT puuid, stats_json FROM player_match_stats WHERE match_id=?",
            (match_id,)
        ).fetchall()
        if not rows:
            self.log(f"[GoldMap] Match {match_id} missing player stats rows")
            return None
        stats = {}
        for puuid, blob in rows:
            data = self._safe_json(blob)
            if not data:
                self.log(f"[GoldMap] Match {match_id} has bad stats JSON for {puuid[:8]}")
                continue
            stats[puuid] = data.get("gold", 0)
        if len(stats) < 10:
            self.log(f"[GoldMap] Match {match_id} has {len(stats)} player stats < 10")
            return None
        self._stats_cache[match_id] = stats
        return stats

    @staticmethod
    def _safe_json(blob):
        try:
            return json.loads(blob)
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    def build_figure(self, match: Match):
        data = self.fetch_data(match)
        if not data["blue"] and not data["red"]:
            print("❌ GoldMap: insufficient shared matches to plot.")
            return None

        fig, ax = plt.subplots(figsize=(7, 6))

        for side, points in data.items():
            if not points:
                continue
            xs = [p["player_share"] for p in points]
            ys = [p["team_share"] for p in points]
            markers = [ROLE_MARKERS.get(p["role"], "o") for p in points]
            for x, y, marker in zip(xs, ys, markers):
                ax.scatter(
                    x,
                    y,
                    s=70,
                    marker=marker,
                    color=TEAM_COLORS.get(side, "#888"),
                    alpha=0.75,
                    edgecolors="k",
                    linewidths=0.3,
                )

        ax.set_xlim(0.05, 0.55)
        ax.set_ylim(0.3, 0.9)
        ax.set_xlabel("Player gold ÷ team gold")
        ax.set_ylabel("Team gold ÷ total gold")
        ax.set_title("Gold Map – Player vs Team Share (last 10 matches)")
        ax.grid(True, linestyle="--", alpha=0.25)

        role_handles = [
            plt.Line2D([0], [0], marker=m, color="w", label=role,
                       markerfacecolor="#777", markeredgecolor="k", markersize=8)
            for role, m in ROLE_MARKERS.items()
        ]
        team_handles = [
            plt.Line2D([0], [0], marker="s", color=color, label=f"{side.title()} team",
                       markerfacecolor=color)
            for side, color in TEAM_COLORS.items()
        ]
        legend1 = ax.legend(handles=team_handles, loc="upper left")
        ax.add_artist(legend1)
        ax.legend(handles=role_handles, title="Role markers", loc="lower right")

        fig.tight_layout()
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