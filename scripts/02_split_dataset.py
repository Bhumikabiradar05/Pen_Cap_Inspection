from pathlib import Path
import random
import shutil
import json
import csv
from PIL import Image
import imagehash

# ---------------- CONFIGURATION ---------------
ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "step1_output" / "unique_images"
DATASET_DIR = ROOT / "dataset"
REPORT_DIR = ROOT / "step1_output" / "reports"
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42
GROUP_PHASH_DISTANCE = 6
SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# ---------------- HELPERS ---------------
def get_phash(path):
    with Image.open(path) as im:
        return imagehash.phash(im.convert("RGB"))

class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

def allocate_groups(group_to_files):
    """Greedy allocation aiming for 70/15/15 while keeping groups intact."""
    rng = random.Random(RANDOM_SEED)
    groups = list(group_to_files.items())
    rng.shuffle(groups)
    groups.sort(key=lambda item: len(item[1]), reverse=True)
    total = sum(len(files) for _, files in groups)
    targets = {
        "train": total * TRAIN_RATIO,
        "validation": total * VALIDATION_RATIO,
        "test": total * TEST_RATIO,
    }
    counts = {"train": 0, "validation": 0, "test": 0}
    assignment = {}
    for gid, files in groups:
        # choose the split with the largest remaining target proportion
        chosen = max(
            counts.keys(),
            key=lambda split: (targets[split] - counts[split]) / max(targets[split], 1)
        )
        assignment[gid] = chosen
        counts[chosen] += len(files)
    return assignment, counts, total

# ---------------- MAIN ---------------
def main():
    if not SOURCE_DIR.exists():
        raise SystemExit(f"Source folder not found: {SOURCE_DIR}. Run Step 1 first.")
    images = sorted(p for p in SOURCE_DIR.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED)
    if len(images) < 3:
        raise SystemExit("Need at least 3 approved images to create three splits.")
    hashes = {}
    for path in images:
        try:
            hashes[path.name] = get_phash(path)
        except Exception as exc:
            print(f"Skipping unreadable image {path.name}: {exc}")
    names = list(hashes.keys())
    if len(names) < 3:
        raise SystemExit("Fewer than 3 readable images remain.")
    # Build similarity groups.
    uf = UnionFind(names)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            if (hashes[a] - hashes[b]) <= GROUP_PHASH_DISTANCE:
                uf.union(a, b)
    group_to_files = {}
    for name in names:
        gid = uf.find(name)
        group_to_files.setdefault(gid, []).append(name)
    if len(group_to_files) < 3:
        raise SystemExit(
            f"Only {len(group_to_files)} similarity groups were found. A three‑way split is unreliable."
        )
    assignment, counts, total = allocate_groups(group_to_files)
    split_dirs = {
        "train": DATASET_DIR / "train" / "good",
        "validation": DATASET_DIR / "validation" / "good",
        "test": DATASET_DIR / "test" / "good",
    }
    for folder in split_dirs.values():
        folder.mkdir(parents=True, exist_ok=True)
    # Ensure output dirs are empty.
    existing = [p for folder in split_dirs.values() for p in folder.iterdir() if p.is_file()]
    if existing:
        raise SystemExit("Dataset output folders are not empty. Clear them before rerunning.")
    rows = []
    for gid, files in group_to_files.items():
        split = assignment[gid]
        for name in files:
            src = SOURCE_DIR / name
            dst = split_dirs[split] / name
            shutil.copy2(src, dst)
            rows.append({
                "filename": name,
                "similarity_group": str(gid),
                "split": split,
                "source": str(src.relative_to(ROOT)),
                "destination": str(dst.relative_to(ROOT)),
            })
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with (REPORT_DIR / "dataset_split.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["filename", "similarity_group", "split", "source", "destination"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    actual = {split: sum(1 for r in rows if r["split"] == split) for split in ["train", "validation", "test"]}
    summary = {
        "approved_images_split": len(rows),
        "similarity_groups": len(group_to_files),
        "target_ratios": {"train": TRAIN_RATIO, "validation": VALIDATION_RATIO, "test": TEST_RATIO},
        "actual_image_counts": actual,
        "actual_percentages": {k: round(v / len(rows) * 100, 2) for k, v in actual.items()},
        "group_phash_distance": GROUP_PHASH_DISTANCE,
        "random_seed": RANDOM_SEED,
        "note": "Groups are kept together; actual ratios may differ from targets."
    }
    (REPORT_DIR / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSTEP 2 COMPLETE")
    print(json.dumps(summary, indent=2))
    print(f"Split report: {REPORT_DIR / 'dataset_split.csv'}")
    print(f"Dataset created at: {DATASET_DIR}")

if __name__ == "__main__":
    main()
