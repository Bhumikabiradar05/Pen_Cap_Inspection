from pathlib import Path
from PIL import Image, ImageFilter, ImageStat
import imagehash
import hashlib
import shutil
import csv
import json

# ---------------- CONFIGURATION ---------------
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "raw_dataset"
OUT_DIR = ROOT / "step1_output"
UNIQUE_DIR = OUT_DIR / "unique_images"
REVIEW_DIR = OUT_DIR / "review_duplicates"
REPORT_DIR = OUT_DIR / "reports"

# pHash Hamming-distance thresholds.
# <= STRICT_DISTANCE: strong duplicate candidate.
# STRICT_DISTANCE+1 through REVIEW_DISTANCE: manual review.
STRICT_DISTANCE = 6
REVIEW_DISTANCE = 12
SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# ---------------- HELPERS ---------------
def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def image_info(path):
    """Return dimensions, pHash, and a rough sharpness score."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        width, height = im.size
        phash = imagehash.phash(im)
        # Variance of edge‑filtered grayscale image: rough sharpness proxy.
        edges = im.convert("L").filter(ImageFilter.FIND_EDGES)
        sharpness = float(ImageStat.Stat(edges).var[0])
    return width, height, str(phash), sharpness

def hash_distance(hash_a, hash_b):
    return imagehash.hex_to_hash(hash_a) - imagehash.hex_to_hash(hash_b)

class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

# ---------------- MAIN ---------------
def main():
    for folder in [UNIQUE_DIR, REVIEW_DIR, REPORT_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in RAW_DIR.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED)
    if not files:
        raise SystemExit(f"No supported images found in {RAW_DIR}. Put image files directly inside raw_dataset.")

    records = {}
    errors = []
    print(f"Found {len(files)} image files. Reading images…")
    for index, path in enumerate(files, start=1):
        try:
            width, height, phash, sharpness = image_info(path)
            records[path.name] = {
                "path": path,
                "sha256": sha256_file(path),
                "width": width,
                "height": height,
                "pixels": width * height,
                "phash": phash,
                "sharpness": sharpness,
            }
        except Exception as exc:
            errors.append({"filename": path.name, "error": str(exc)})
        print(f"Processed {index}/{len(files)}: {path.name}")

    names = list(records.keys())
    uf = UnionFind(names)
    pair_rows = []
    borderline_names = set()

    # Compare every pair.
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            ra, rb = records[a], records[b]
            if ra["sha256"] == rb["sha256"]:
                distance = 0
                match_type = "exact_duplicate"
            else:
                distance = hash_distance(ra["phash"], rb["phash"])
                if distance <= STRICT_DISTANCE:
                    match_type = "strict_near_duplicate"
                elif distance <= REVIEW_DISTANCE:
                    match_type = "borderline_review"
                else:
                    continue
            pair_rows.append({
                "image_1": a,
                "image_2": b,
                "phash_distance": distance,
                "match_type": match_type,
            })
            if distance <= STRICT_DISTANCE:
                uf.union(a, b)
            else:
                borderline_names.update([a, b])

    # Build strict duplicate clusters.
    clusters = {}
    for name in names:
        clusters.setdefault(uf.find(name), []).append(name)

    strict_duplicate_names = set()
    representatives = set()
    for members in clusters.values():
        best = max(members, key=lambda n: (records[n]["pixels"], records[n]["sharpness"]))
        representatives.add(best)
        for name in members:
            if name != best:
                strict_duplicate_names.add(name)

    review_names = strict_duplicate_names | (borderline_names - strict_duplicate_names)
    unique_names = set(names) - review_names

    audit_rows = []
    for name in names:
        rec = records[name]
        if name in strict_duplicate_names:
            decision = "review_duplicate_of_cluster_representative"
            destination = REVIEW_DIR / name
        elif name in borderline_names:
            decision = "manual_review_borderline_similarity"
            destination = REVIEW_DIR / name
        else:
            decision = "provisionally_unique_visual_check_required"
            destination = UNIQUE_DIR / name
        shutil.copy2(rec["path"], destination)
        audit_rows.append({
            "filename": name,
            "sha256": rec["sha256"],
            "width": rec["width"],
            "height": rec["height"],
            "sharpness_proxy": round(rec["sharpness"], 3),
            "phash": rec["phash"],
            "decision": decision,
            "copied_to": str(destination.relative_to(ROOT)),
        })

    # CSV reports
    with (REPORT_DIR / "image_audit.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
        writer.writeheader()
        writer.writerows(audit_rows)
    with (REPORT_DIR / "similarity_pairs.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["image_1", "image_2", "phash_distance", "match_type"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pair_rows)
    with (REPORT_DIR / "read_errors.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["filename", "error"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(errors)

    summary = {
        "raw_image_files": len(files),
        "successfully_read": len(records),
        "read_errors": len(errors),
        "provisional_unique_copies": len(unique_names),
        "review_copies": len(review_names),
        "similar_pairs_found": len(pair_rows),
        "strict_distance": STRICT_DISTANCE,
        "borderline_distance_upper_limit": REVIEW_DISTANCE,
        "warning": "Review all borderline and duplicate candidates manually before approving the final dataset."
    }
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSTEP 1 AUDIT COMPLETE")
    print(json.dumps(summary, indent=2))
    print(f"\nOpen this report: {REPORT_DIR / 'summary.json'}")
    print(f"Review image pairs: {REPORT_DIR / 'similarity_pairs.csv'}")
    print("No original raw images were deleted.")

if __name__ == "__main__":
    main()
