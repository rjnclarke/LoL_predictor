import base64
import io
import sqlite3
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import streamlit as st

from core.entities import Match, Player, Team
from crawler.match_base import MatchBase
from viz.gold_contribution import GoldContribution
from viz.gold_map import GoldMap
from viz.player_profile import PlayerProfile
from viz.spider_stats import SpiderStats
from viz.nn_infer import predict_match_outcome, build_speedometer

DB_DEFAULT_PATH = "data/match_base/live.db"

APP_STYLES = """
<style>
.stApp {background: radial-gradient(circle at 12% 20%, rgba(59,130,246,0.25), transparent 35%),
           radial-gradient(circle at 88% 10%, rgba(248,113,113,0.18), transparent 30%),
           #050914; color:#f8fafc;}
.block-container {max-width: 1100px; padding-top:1.5rem;}
.hero {background: linear-gradient(120deg, rgba(56,189,248,0.2), rgba(248,113,113,0.2));
    border:1px solid rgba(148,163,184,0.25); border-radius:22px; padding:1.25rem 1.5rem;
    box-shadow:0 25px 60px rgba(2,6,23,0.55); margin-bottom:1.7rem;}
.hero h1 {margin-bottom:0.3rem; font-size:2.05rem; font-weight:600;}
.viz-card {background: rgba(15,23,42,0.9); border-radius:20px; padding:1.2rem 1.3rem;
       border:1px solid rgba(148,163,184,0.18); margin-bottom:1.2rem;
       box-shadow:0 18px 35px rgba(2,6,23,0.45);}
.viz-card h3 {margin-bottom:0.4rem; font-size:1.05rem; letter-spacing:0.03em;}
.viz-desc {margin:0 0 0.65rem; color:rgba(226,232,240,0.8); font-size:0.9rem;}
.team-panel {background: rgba(15,23,42,0.85); border-radius:18px; padding:1rem;
         border:1px solid rgba(148,163,184,0.18); margin-bottom:1.2rem;}
.team-panel h3 {margin-bottom:0.5rem; font-size:1.05rem; text-transform:uppercase; letter-spacing:0.08em;}
</style>
"""

PLAYER_POPOVER_STYLES = """
<style>
.player-pill-row {display:flex; flex-wrap:wrap; gap:0.5rem; margin-bottom:0.75rem;}
.player-pill {position:relative; padding:0.35rem 0.75rem; border-radius:999px; background:#111827; color:#f3f4f6; font-size:0.85rem; cursor:pointer;}
.player-popover {display:none; position:absolute; top:110%; left:50%; transform:translateX(-50%); z-index:5; background:#111827; padding:0.5rem; border-radius:0.5rem; box-shadow:0 8px 24px rgba(0,0,0,0.25);}
.player-pill:hover .player-popover {display:block;}
.player-popover img {width:360px; max-width:70vw; border-radius:0.35rem;}
.player-popover-empty {color:#9ca3af; font-size:0.8rem;}
</style>
"""


def inject_styles():
    st.markdown(APP_STYLES + PLAYER_POPOVER_STYLES, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_match_base(db_path: str) -> MatchBase:
    """Cache a MatchBase wrapper so we reuse the same SQLite connection."""
    return MatchBase(live_path=db_path, update_path=f"{db_path}.tmp")


@st.cache_data(show_spinner=False)
def load_players(db_path: str):
    """Return ordered player metadata (most recently scraped first)."""
    if not Path(db_path).exists():
        return []
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT puuid, tier, last_scraped FROM players ORDER BY last_scraped DESC"
        ).fetchall()
    return rows


def fig_to_base64(fig) -> str:
    """Convert a Matplotlib figure to a data URI string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def render_player_popovers(player_profile: PlayerProfile, match: Match, label_map: dict[str, str]):
    """Render hoverable pills that reveal player profile plots."""
    for side_label, team in (("Blue", match.blue), ("Red", match.red)):
        st.markdown(f"**{side_label} side**")
        pills = []
        for player in team.players:
            fig = player_profile.build_figure(player)
            if fig:
                img_src = fig_to_base64(fig)
                inner = f"<img src='data:image/png;base64,{img_src}' alt='Profile {player.puuid[:8]}'>"
            else:
                inner = "<div class='player-popover-empty'>No profile data</div>"
            label = label_map.get(player.puuid, player.puuid[:12])
            pills.append(
                f"<div class='player-pill'>{label}<div class='player-popover'>{inner}</div></div>"
            )
        st.markdown(
            f"<div class='player-pill-row'>{''.join(pills)}</div>",
            unsafe_allow_html=True,
        )


def render_card(title: str, body_cb: Callable[[], None], description: str | None = None):
    extra = f"<p class='viz-desc'>{description}</p>" if description else ""
    st.markdown(f"<div class='viz-card'><h3>{title}</h3>{extra}", unsafe_allow_html=True)
    body_cb()
    st.markdown("</div>", unsafe_allow_html=True)


def format_player_option(option: str, label_map: dict[str, str]) -> str:
    if not option:
        return "— select player —"
    return label_map.get(option, option[:12])


def current_team_selection(side: str) -> list[str]:
    return [st.session_state.get(f"{side}_{role}", "") for role in Team.ROLES_ORDER]


def build_team_selector(
    side: str,
    player_ids: list[str],
    label_map: dict[str, str],
    exclude_other: set[str] | None = None,
):
    selections = []
    used: set[str] = set()
    exclude_other = exclude_other or set()

    for role in Team.ROLES_ORDER:
        remaining = [
            pid for pid in player_ids
            if pid not in used and pid not in exclude_other
        ]
        options = [""] + remaining
        key = f"{side}_{role}"
        current = st.session_state.get(key, "")
        if current and current not in options:
            st.session_state[key] = ""
            current = ""
        index = options.index(current) if current in options else 0
        choice = st.selectbox(
            f"{side.title()} {role}",
            options,
            index=index,
            format_func=lambda opt, lm=label_map: format_player_option(opt, lm),
            key=key,
        )
        if choice:
            used.add(choice)
        selections.append(choice)
    return [sel for sel in selections if sel]


def build_match(blue_ids, red_ids):
    try:
        blue_team = Team([Player(pid) for pid in blue_ids])
        red_team = Team([Player(pid) for pid in red_ids])
        return Match(blue_team, red_team)
    except ValueError:
        return None


def render_viz(match_base: MatchBase, match: Match, label_map: dict[str, str]):
    inference = predict_match_outcome(match_base, match)

    def draw_inference():
        gauge_fig = build_speedometer(inference["blue_success"])
        st.pyplot(gauge_fig)
        plt.close(gauge_fig)
        st.caption(
            f"Raw {inference['raw']:.3f} → Blue success {inference['blue_success']*100:,.1f}% | "
            f"Red {inference['red_success']*100:,.1f}%"
        )

    render_card("Neural inference (blue ↔ red tilt)", draw_inference)

    def draw_gold_contribution():
        gold_viz = GoldContribution(match_base)
        gold_fig = gold_viz.build_figure(match)
        if gold_fig:
            st.pyplot(gold_fig)
            plt.close(gold_fig)
        else:
            st.info("Gold contribution figure unavailable for the selected players.")

    render_card(
        "Gold contribution (team normalized)",
        draw_gold_contribution,
        "Shows how much of their own team's gold each player secured over the last ten games.",
    )

    def draw_gold_map():
        gold_map_viz = GoldMap(match_base)
        gold_map_fig = gold_map_viz.build_figure(match)
        if gold_map_fig:
            st.pyplot(gold_map_fig)
            plt.close(gold_map_fig)
        else:
            st.info("No overlapping matches to render the gold map heatmaps.")

    render_card(
        "Gold map density",
        draw_gold_map,
        "Wide heat bands = volatile team performance; bright concentrated heat = consistent gold control.",
    )

    def draw_spider():
        spider_viz = SpiderStats(match_base)
        spider_fig = spider_viz.build_figure(match)
        if spider_fig:
            st.pyplot(spider_fig)
            plt.close(spider_fig)
        else:
            st.info("Not enough historical stats to build the spider chart.")

    render_card("Spider stats (team comparison)", draw_spider)

    def draw_profiles():
        player_profile_viz = PlayerProfile(match_base)
        render_player_popovers(player_profile_viz, match, label_map)

    render_card("Player profiles (hover for detail)", draw_profiles)


def main():
    inject_styles()
    st.markdown(
        "<div class='hero'>"
        "<h1>LoL Match Visualizer</h1>"
        "<p>Lock in scrim rosters, surface historical gold trends, hover player cards,"
        " and let the NN tell you which side it trusts more.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    db_path = DB_DEFAULT_PATH
    if not Path(db_path).exists():
        st.error(f"Database not found at {db_path}")
        return
    st.caption(f"Using MatchBase at `{db_path}`")

    match_base = load_match_base(db_path)
    players = load_players(db_path)
    if not players:
        st.warning("No players found. Seed or update the database first.")
        return

    match_count = match_base.db.conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]

    stats_cols = st.columns(3)
    stats_cols[0].metric("Players tracked", len(players))
    stats_cols[1].metric("Matches stored", match_count)
    stats_cols[2].metric("DB", "Live cache")

    player_ids = [row[0] for row in players]
    label_map = {
        row[0]: f"{row[0][:12]}… ({row[1] or 'tier ?'})" for row in players
    }

    st.markdown("### Draft teams")
    cols = st.columns(2, gap="large")
    red_taken = {pid for pid in current_team_selection("red") if pid}
    with cols[0]:
        st.markdown("<div class='team-panel'><h3>Blue side</h3>", unsafe_allow_html=True)
        blue_ids = build_team_selector("blue", player_ids, label_map, exclude_other=red_taken)
        st.markdown("</div>", unsafe_allow_html=True)
    with cols[1]:
        st.markdown("<div class='team-panel'><h3>Red side</h3>", unsafe_allow_html=True)
        red_ids = build_team_selector("red", player_ids, label_map, exclude_other=set(blue_ids))
        st.markdown("</div>", unsafe_allow_html=True)

    if len(blue_ids) != 5 or len(red_ids) != 5:
        st.info("Select five distinct players for both teams to continue.")
        return

    if set(blue_ids) & set(red_ids):
        st.warning("A player cannot be on both teams. Adjust your selections.")
        return

    match = build_match(blue_ids, red_ids)
    if not match:
        st.error("Unable to build the match with the chosen players.")
        return

    st.success("Teams locked in! Rendering visualizations…")
    render_viz(match_base, match, label_map)


if __name__ == "__main__":
    main()
