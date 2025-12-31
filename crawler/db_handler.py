import sqlite3, time, json

class DatabaseHandler:
    """
    Lightweight SQLite layer for players, matches, and per‑match stats.

    S – Keeps only persistence logic (Single Responsibility).
    O – Open for extension: we added player_match_stats without changing
        existing behaviour used by earlier components.
    """

    def __init__(self, path="data/matches.db"):
        # connect allows multiple threads via check_same_thread=False if async tasks later write
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self._create_core_tables()
        self._apply_migrations()
        self._create_indexes()
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # --- schema helpers ------------------------------------------------#
    # ------------------------------------------------------------------ #
    def _create_core_tables(self):
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS players (
            puuid TEXT PRIMARY KEY,
            tier TEXT,
            discovered INTEGER DEFAULT 0,
            in_match INTEGER DEFAULT 0,
            has_features INTEGER DEFAULT 0,
            last_scraped REAL
        );

        CREATE TABLE IF NOT EXISTS matches (
            match_id TEXT PRIMARY KEY,
            timestamp REAL,
            scraped_at REAL,
            puuids_json TEXT,
            label REAL,
            vector_complete INTEGER DEFAULT 0,
            winner_side TEXT,
            blue_gold REAL,
            red_gold REAL,
            player_gold_json TEXT
        );

        CREATE TABLE IF NOT EXISTS player_match_stats (
            puuid TEXT,
            match_id TEXT,
            timestamp REAL,
            role TEXT,
            stats_json TEXT,
            PRIMARY KEY (puuid, match_id)
        );

        CREATE TABLE IF NOT EXISTS player_features (
            puuid TEXT PRIMARY KEY,
            tier_norm REAL,
            static_json TEXT,
            dynamic_json TEXT,
            games_used INTEGER,
            last_updated REAL
        );
        """)

    def _apply_migrations(self):
        self._ensure_column(
            table="players",
            column="has_features",
            definition="INTEGER DEFAULT 0"
        )
        self._ensure_column(
            table="matches",
            column="vector_complete",
            definition="INTEGER DEFAULT 0"
        )
        self._ensure_column(
            table="matches",
            column="winner_side",
            definition="TEXT"
        )
        self._ensure_column(
            table="matches",
            column="blue_gold",
            definition="REAL"
        )
        self._ensure_column(
            table="matches",
            column="red_gold",
            definition="REAL"
        )
        self._ensure_column(
            table="matches",
            column="player_gold_json",
            definition="TEXT"
        )

    def _ensure_column(self, table, column, definition):
        cur = self.conn.execute(f"PRAGMA table_info({table})")
        columns = {row[1] for row in cur.fetchall()}
        if column not in columns:
            self.conn.execute(
                f"ALTER TABLE {table} ADD COLUMN {column} {definition}"
            )

    def _create_indexes(self):
        self.conn.executescript("""
        CREATE INDEX IF NOT EXISTS idx_player_time
            ON player_match_stats (puuid, timestamp);
        CREATE INDEX IF NOT EXISTS idx_match_id
            ON player_match_stats (match_id);
        CREATE INDEX IF NOT EXISTS idx_match_complete
            ON matches (vector_complete);
        CREATE INDEX IF NOT EXISTS idx_features_puuid
            ON player_features (puuid);
        """)

    # ------------------------------------------------------------------ #
    # --- player helpers ------------------------------------------------#
    # ------------------------------------------------------------------ #
    def insert_player(self, puuid, tier=None, discovered=0):
        self.conn.execute(
            "INSERT OR IGNORE INTO players(puuid, tier, discovered) VALUES(?,?,?)",
            (puuid, tier, discovered)
        )
        self.conn.commit()

    def player_batches(self, limit=10):
        # SQLite uses IS NULL, not 'NULLS FIRST' syntax
        cur = self.conn.execute(
            "SELECT puuid FROM players ORDER BY last_scraped IS NULL DESC, last_scraped ASC LIMIT ?",
            (limit,)
        )
        return [r[0] for r in cur.fetchall()]

    def mark_scraped(self, puuid):
        self.conn.execute(
            "UPDATE players SET last_scraped=? WHERE puuid=?",
            (time.time(), puuid)
        )
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # --- match helpers -------------------------------------------------#
    # ------------------------------------------------------------------ #
    def match_exists(self, mid):
        cur = self.conn.execute("SELECT 1 FROM matches WHERE match_id=?", (mid,))
        return cur.fetchone() is not None

    def insert_match(self, mid, info, ordered_puuids, label):
        ts = info.get("gameStartTimestamp", time.time())
        teams = info.get("teams", []) or []
        participants = info.get("participants", []) or []
        player_gold = {
            p.get("puuid"): p.get("goldEarned", 0)
            for p in participants if p.get("puuid")
        }
        blue_gold = sum(p.get("goldEarned", 0) for p in participants if p.get("teamId") == 100)
        red_gold = sum(p.get("goldEarned", 0) for p in participants if p.get("teamId") == 200)
        winner_side = None
        for team in teams:
            tid = team.get("teamId")
            if team.get("win") and tid == 100:
                winner_side = "blue"
                break
            if team.get("win") and tid == 200:
                winner_side = "red"
                break

        self.conn.execute("""
            INSERT OR REPLACE INTO matches
                (match_id, timestamp, scraped_at, puuids_json, label,
                 winner_side, blue_gold, red_gold, player_gold_json)
            VALUES(?,?,?,?,?,?,?,?,?)
        """, (
            mid,
            ts,
            time.time(),
            json.dumps(ordered_puuids),
            label,
            winner_side,
            blue_gold,
            red_gold,
            json.dumps(player_gold),
        ))
        self.conn.commit()

    def match_count(self):
        cur = self.conn.execute("SELECT COUNT(*) FROM matches")
        return cur.fetchone()[0]

    # ------------------------------------------------------------------ #
    # --- player_match_stats helpers -----------------------------------#
    # ------------------------------------------------------------------ #
    def insert_player_match(self, puuid, match_id, timestamp, role, stats_dict):
        """
        Insert or replace one player's stats for one match.
        Encapsulates JSON‑serialization and commit for clarity.
        """
        self.conn.execute("""
            INSERT OR REPLACE INTO player_match_stats
                (puuid, match_id, timestamp, role, stats_json)
            VALUES (?,?,?,?,?)
        """, (puuid, match_id, timestamp, role, json.dumps(stats_dict)))
        self.conn.commit()

    def get_recent_matches(self, puuid, limit=10):
        """Return latest <limit> matches for a player as list of (match_id, stats_json)."""
        cur = self.conn.execute("""
            SELECT match_id, stats_json, role, timestamp
            FROM player_match_stats
            WHERE puuid=?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (puuid, limit))
        return cur.fetchall()

    def delete_old_matches(self, puuid, keep=10):
        """Slide‑window cleanup: delete all but <keep> most recent matches."""
        self.conn.execute("""
            DELETE FROM player_match_stats
            WHERE rowid NOT IN (
                SELECT rowid FROM player_match_stats
                WHERE puuid=? ORDER BY timestamp DESC LIMIT ?
            )
            AND puuid=?;
        """, (puuid, keep, puuid))
        self.conn.commit()