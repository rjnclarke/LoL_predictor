import json, random, sqlite3, torch
from torch.utils.data import Dataset, random_split


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def safe_num(x):
    try:
        if x is None or (isinstance(x, float) and torch.isnan(torch.tensor(x))):
            return 0.0
        return float(x)
    except Exception:
        return 0.0


def build_player_vector(static_json, dynamic_json):
    """Combine static+dynamic JSON into 13‑feature numeric vector."""
    s = json.loads(static_json)
    d = json.loads(dynamic_json)
    rank_map = {"IV": 1, "III": 2, "II": 3, "I": 4}
    rank_strength = safe_num(rank_map.get(s.get("rank"), 0)) * safe_num(
        s.get("league_points")
    )

    feats = [
        safe_num(s.get("tier_norm")),
        rank_strength,
        safe_num(s.get("summoner_level")),
        safe_num(s.get("mastery_score")),
        safe_num(s.get("challenge_points")),
        safe_num(d.get("kills_avg")),
        safe_num(d.get("deaths_avg")),
        safe_num(d.get("assists_avg")),
        safe_num(d.get("gold_per_min")),
        safe_num(d.get("cs_per_min")),
        safe_num(d.get("vision_score")),
        safe_num(d.get("damage_to_champs")),
        safe_num(d.get("win_rate_recent")),
    ]
    return feats


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class MatchDataset(Dataset):
    """
    PyTorch dataset returning per‑match tensors:
        X : Tensor[10, 13] (player features)
        y : Tensor[1]      (match label)
    """

    def __init__(self, db_path, seed=42):
        self.conn = sqlite3.connect(db_path)
        self.samples = self._load_all_matches()
        self.conn.close()

        # shuffle right after loading
        random.seed(seed)
        random.shuffle(self.samples)

    def _load_all_matches(self):
        rows = self.conn.execute(
            "SELECT match_id, puuids_json, label, vector_complete FROM matches"
        ).fetchall()

        samples = []
        for match_id, puuids_json, label, complete in rows:
            if not complete:
                continue
            try:
                puuids = json.loads(puuids_json)
            except Exception:
                continue
            if len(puuids) != 10:
                continue

            all_vecs = []
            skip = False
            for puuid in puuids:
                row = self.conn.execute(
                    "SELECT static_json, dynamic_json FROM player_features WHERE puuid=?",
                    (puuid,),
                ).fetchone()
                if not row:
                    skip = True
                    break
                all_vecs.append(build_player_vector(*row))
            if skip or len(all_vecs) != 10:
                continue

            X = torch.tensor(all_vecs, dtype=torch.float32)
            y = torch.tensor([safe_num(label)], dtype=torch.float32)
            samples.append((X, y))

        print(f"Loaded {len(samples)} complete matches.")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ---------------------------------------------------------------------
# Split & Save
# ---------------------------------------------------------------------

def prepare_splits(db_path, out_dir="data", seed=42):
    ds = MatchDataset(db_path, seed=seed)
    total = len(ds)
    n_train = int(total * 0.8)
    n_dev = int(total * 0.1)
    n_test = total - n_train - n_dev

    torch.manual_seed(seed)
    train, dev, test = random_split(ds, [n_train, n_dev, n_test])

    def save_split(name, split):
        X = torch.stack([x for x, _ in split])
        y = torch.stack([y for _, y in split])
        cpu_path = f"{out_dir}/{name}_cpu.pt"
        torch.save((X, y), cpu_path)
        if torch.cuda.is_available():
            gpu_path = f"{out_dir}/{name}_gpu.pt"
            torch.save((X.to("cuda"), y.to("cuda")), gpu_path)
        print(f"Saved {name}: {len(split)} matches")

    save_split("train", train)
    save_split("dev", dev)
    save_split("test", test)
    print(f"✅ All splits saved to {out_dir}.")


if __name__ == "__main__":
    prepare_splits("data/matches.db")