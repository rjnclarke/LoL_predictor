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
        cur = self.conn.cursor()

        # Core tables
        cur.executescript("""
        CREATE TABLE IF NOT EXISTS players (
            puuid TEXT PRIMARY KEY,
            tier TEXT,
            discovered INTEGER DEFAULT 0,
            in_match INTEGER DEFAULT 0,
            last_scraped REAL
        );

        CREATE TABLE IF NOT EXISTS matches (
            match_id TEXT PRIMARY KEY,
            timestamp REAL,
            scraped_at REAL,
            puuids_json TEXT,
            label REAL
        );

        /* New per‑player × match stats table.
           Each row tracks one player's performance in one match.
           Stored in long format (easier to query + extend). */
        CREATE TABLE IF NOT EXISTS player_match_stats (
            puuid TEXT,
            match_id TEXT,
            timestamp REAL,
            role TEXT,
            stats_json TEXT,
            PRIMARY KEY (puuid, match_id)
        );

        /* Indices for fast lookups */
        CREATE INDEX IF NOT EXISTS idx_player_time
            ON player_match_stats (puuid, timestamp);
        CREATE INDEX IF NOT EXISTS idx_match_id
            ON player_match_stats (match_id);
        """)
        self.conn.commit()

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
        self.conn.execute("""
            INSERT OR IGNORE INTO matches(match_id, timestamp, scraped_at, puuids_json, label)
            VALUES(?,?,?,?,?)
        """, (mid, ts, time.time(), json.dumps(ordered_puuids), label))
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