"""GoldMap density heatmap visualization."""

from .base_viz import ABCViz
from core.entities import Match, Team
import json
import numpy as np
import matplotlib.pyplot as plt

PLAYER_SHARE_RANGE = (0.05, 0.6)
TEAM_SHARE_RANGE = (0.35, 0.85)
HEATMAP_BINS = (40, 24)


class GoldMap(ABCViz):
    """Twin density heatmaps showing where gold share combos cluster."""

    def __init__(self, match_base):
        super().__init__(match_base)
        self._match_cache = {}
        self._stats_cache = {}

    # ------------------------------------------------------------------ #
    def fetch_data(self, match: Match, per_player: int = 30):
        """Return player/team gold share samples for both sides."""
        payload = {"blue": [], "red": []}
        roster = {
            "blue": list(zip(Team.ROLES_ORDER, match.blue.players)),
            "red": list(zip(Team.ROLES_ORDER, match.red.players)),
        }

        for side, entries in roster.items():
            for role, player in entries:
                rows = self.conn.execute(
                    """
                    SELECT match_id, stats_json
                    FROM player_match_stats
                    WHERE puuid=?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (player.puuid, per_player)
                ).fetchall()

                for match_id, stats_blob in rows:
                    stats = self._safe_json(stats_blob)
                    if not stats:
                        continue
                    share = self._compute_shares(match_id, player.puuid, stats.get("gold", 0))
                    if not share:
                        continue
                    payload[side].append(
                        {
                            "player_share": share["player_share"],
                            "team_share": share["team_share"],
                            "role": role,
                        }
                    )

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

        fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
        sides = [("blue", plt.cm.Blues), ("red", plt.cm.Reds)]

        for ax, (side, cmap) in zip(axes, sides):
            samples = data[side]
            if not samples:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
                ax.set_axis_off()
                continue
            player_shares = np.clip([s["player_share"] for s in samples], *PLAYER_SHARE_RANGE)
            team_shares = np.clip([s["team_share"] for s in samples], *TEAM_SHARE_RANGE)
            heat, xedges, yedges = np.histogram2d(
                player_shares,
                team_shares,
                bins=HEATMAP_BINS,
                range=(PLAYER_SHARE_RANGE, TEAM_SHARE_RANGE),
            )
            heat = np.log1p(heat)
            if heat.max() > 0:
                heat /= heat.max()
            im = ax.imshow(
                heat.T,
                origin="lower",
                extent=(
                    PLAYER_SHARE_RANGE[0],
                    PLAYER_SHARE_RANGE[1],
                    TEAM_SHARE_RANGE[0],
                    TEAM_SHARE_RANGE[1],
                ),
                aspect="auto",
                cmap=cmap,
            )
            ax.set_title(f"{side.title()} density of gold share")
            ax.set_xlabel("Player gold ÷ team gold")
            ax.set_ylabel("Team gold ÷ total gold")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Relative density")
            ax.grid(True, color="white", alpha=0.15, linewidth=0.4)

        fig.suptitle("Gold Map – high resolution gold-share heatmaps", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
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