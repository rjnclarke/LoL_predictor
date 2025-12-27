import sqlite3, time, json

class DatabaseHandler:
    """Tiny SQLite layer for players & matches persistence."""

    def __init__(self, path="data/matches.db"):
        self.conn = sqlite3.connect(path)
        cur = self.conn.cursor()
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
        """)
        self.conn.commit()

    # --- player helpers ---
    def insert_player(self, puuid, tier=None, discovered=0):
        self.conn.execute(
            "INSERT OR IGNORE INTO players(puuid, tier, discovered) VALUES(?,?,?)",
            (puuid, tier, discovered),
        )
        self.conn.commit()

    def player_batches(self, limit=10):
        cur = self.conn.execute(
            "SELECT puuid FROM players ORDER BY last_scraped NULLS FIRST LIMIT ?", (limit,)
        )
        return [r[0] for r in cur.fetchall()]

    def mark_scraped(self, puuid):
        self.conn.execute("UPDATE players SET last_scraped=? WHERE puuid=?",
                          (time.time(), puuid))
        self.conn.commit()

    # --- match helpers ---
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