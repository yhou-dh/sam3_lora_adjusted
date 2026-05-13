"""
pair_data.py
Pairs segmented illustration crops with text and metadata from CSV files.

Naming convention:
  CSV imgID   : bdj_qm_5_1
  Image file  : bdj_qm_5_il_1.jpg  (inserts _il_ before the last number)

Outputs:
  - image_text_pairs.json / .csv        — image path + text only
  - image_text_enriched.json / .csv     — image path + text + all CSV fields

Usage:
    # Single CSV, combined output
    python3 pair_data.py \
        --csv_dir data/csv \
        --predictions_root predictions/lora \
        --extraction_type mask \
        --output_dir pairs/output \
        --mode combined

    # Per-book output
    python3 pair_data.py \
        --csv_dir data/csv \
        --predictions_root predictions/lora \
        --extraction_type mask \
        --output_dir pairs/output \
        --mode per_book

    # Both combined and per-book
    python3 pair_data.py \
        --csv_dir data/csv \
        --predictions_root predictions/lora \
        --extraction_type mask \
        --output_dir pairs/output \
        --mode both
"""

import argparse
import csv
import json
import os
import re
from pathlib import Path
from collections import defaultdict


# ── Naming conversion ─────────────────────────────────────────────────────────

def imgid_to_filename(img_id: str) -> list:
    """
    Convert CSV imgID to possible image filenames.

    bdj_qm_5_1  →  bdj_qm_5_il_1.jpg / .png
    jx_mg_vol10_15_1 → jx_mg_vol10_15_il_1.jpg / .png

    Strategy: insert _il_ before the last _N segment.
    Returns list of candidate filenames (multiple extensions).
    """
    # Match last _NUMBER at end of string
    m = re.match(r'^(.+)_(\d+)$', img_id)
    if not m:
        return []

    base   = m.group(1)  # e.g. bdj_qm_5
    suffix = m.group(2)  # e.g. 1

    candidates = []
    for ext in ['.jpg', '.jpeg', '.png']:
        candidates.append(f"{base}_il_{suffix}{ext}")
    return candidates


# ── CSV loading ───────────────────────────────────────────────────────────────

def load_csvs(csv_dir: Path) -> list:
    """
    Load all CSVs from a directory.
    Returns list of dicts, one per row, with a '_source_csv' field added.
    Handles BOM and varying columns gracefully.
    """
    all_rows = []
    csv_files = sorted(csv_dir.glob("*.csv"))

    if not csv_files:
        print(f"⚠ No CSV files found in {csv_dir}")
        return []

    for csv_path in csv_files:
        print(f"  Loading {csv_path.name}...")
        with open(csv_path, encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row['_source_csv'] = csv_path.name
                all_rows.append(row)
        print(f"    → {sum(1 for r in all_rows if r['_source_csv'] == csv_path.name)} rows")

    print(f"  Total rows loaded: {len(all_rows)}")
    return all_rows


# ── Image index ───────────────────────────────────────────────────────────────

def build_image_index(image_dir: Path) -> dict:
    """
    Build a lookup: filename (lowercase, no ext) → full path.
    """
    index = {}
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    for p in image_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            index[p.name.lower()] = p
    print(f"  Indexed {len(index)} images in {image_dir}")
    return index


def find_image(img_id: str, image_index: dict) -> Path | None:
    """Try all candidate filenames for a given imgID."""
    for candidate in imgid_to_filename(img_id):
        match = image_index.get(candidate.lower())
        if match:
            return match
    return None


# ── Pairing ───────────────────────────────────────────────────────────────────

def build_pairs(rows: list, image_index: dict, image_dir: Path) -> tuple:
    """
    Returns:
      pairs        — list of {image_path, text} dicts
      enriched     — list of {image_path, text, ...all csv fields} dicts
      stats        — match/miss counts
    """
    pairs    = []
    enriched = []
    matched  = 0
    missed   = 0
    missed_ids = []

    for row in rows:
        img_id = row.get('imgID', '').strip()
        text   = row.get('original_text', '').strip()

        if not img_id:
            continue

        img_path = find_image(img_id, image_index)

        if img_path is None:
            missed += 1
            missed_ids.append(img_id)
            continue

        matched += 1
        rel_path = str(img_path.relative_to(image_dir.parent)
                       if image_dir.parent in img_path.parents
                       else img_path)

        # Basic pair
        pairs.append({
            'image_path': rel_path,
            'text':       text,
        })

        # Enriched pair — all CSV fields except internal ones
        enriched_row = {'image_path': rel_path, 'text': text}
        for k, v in row.items():
            if k not in ('imgID', 'original_text', '_source_csv'):
                enriched_row[k] = v
        enriched_row['source_csv'] = row.get('_source_csv', '')
        enriched.append(enriched_row)

    stats = {
        'matched':    matched,
        'missed':     missed,
        'missed_ids': missed_ids[:20],  # show first 20 only
    }
    return pairs, enriched, stats


# ── Output writers ────────────────────────────────────────────────────────────

def write_json(data: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  ✅ JSON → {path} ({len(data)} records)")


def write_csv(data: list, path: Path):
    if not data:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(data[0].keys())
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    print(f"  ✅ CSV  → {path} ({len(data)} records)")


def save_outputs(pairs: list, enriched: list, output_dir: Path,
                 prefix: str = ""):
    tag = f"{prefix}_" if prefix else ""
    write_json(pairs,    output_dir / f"{tag}image_text_pairs.json")
    write_csv( pairs,    output_dir / f"{tag}image_text_pairs.csv")
    write_json(enriched, output_dir / f"{tag}image_text_enriched.json")
    write_csv( enriched, output_dir / f"{tag}image_text_enriched.csv")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir",    required=True,
                        help="Folder containing CSV files")
    parser.add_argument("--predictions_root", default="predictions/lora",
                        help="Predictions root folder containing book subfolders "
                             "(default: predictions/lora)")
    parser.add_argument("--extraction_type", choices=["mask", "bbox"], default="mask",
                        help="Use mask (foreground/) or bbox crops (default: mask)")
    parser.add_argument("--output_dir", default="pairs/output",
                        help="Output folder (default: outputs/pairs)")
    parser.add_argument("--mode",       choices=["combined", "per_book", "both"],
                        default="combined",
                        help="Output mode: combined (one file), per_book, or both (default: combined)")
    args = parser.parse_args()

    csv_dir    = Path(args.csv_dir)
    pred_root  = Path(args.predictions_root)
    output_dir = Path(args.output_dir)

    # Build image_dir based on extraction type
    subdir = "foreground" if args.extraction_type == "mask" else "bbox"

    print(f"\n{'='*60}")
    print(f"Pairing images with text data")
    print(f"  CSV dir          : {csv_dir}")
    print(f"  Predictions root : {pred_root}")
    print(f"  Extraction type  : {args.extraction_type} ({subdir}/illustration/)")
    print(f"  Output           : {output_dir}")
    print(f"  Mode             : {args.mode}")
    print(f"{'='*60}\n")

    # Load all CSVs
    print("Loading CSVs...")
    all_rows = load_csvs(csv_dir)
    if not all_rows:
        return

    # Build image index — search all <bookid>/<subdir>/illustration/ under pred_root
    print("\nIndexing images...")
    image_index = {}
    search_dirs = sorted(pred_root.glob(f"*/{subdir}/illustration"))
    if not search_dirs:
        print(f"❌ No illustration folders found under {pred_root}/*/{subdir}/illustration/")
        return
    for d in search_dirs:
        image_index.update(build_image_index(d))
    if not image_index:
        print(f"❌ No images found")
        return

    # Combined output
    if args.mode in ("combined", "both"):
        print("\nBuilding combined pairs...")
        pairs, enriched, stats = build_pairs(all_rows, image_index, pred_root)
        print(f"  Matched : {stats['matched']}")
        print(f"  Missed  : {stats['missed']}")
        if stats['missed_ids']:
            print(f"  Sample unmatched IDs: {stats['missed_ids'][:5]}")
        save_outputs(pairs, enriched, output_dir)

    # Per-book output
    if args.mode in ("per_book", "both"):
        print("\nBuilding per-book pairs...")

        # Group rows by book (derived from pageID prefix)
        book_rows = defaultdict(list)
        for row in all_rows:
            page_id = row.get('pageID', '')
            # Book name = everything before the last _NUMBER
            m = re.match(r'^(.+)_\d+$', page_id)
            book = m.group(1) if m else page_id
            book_rows[book].append(row)

        for book, rows in sorted(book_rows.items()):
            print(f"\n  Book: {book} ({len(rows)} rows)")
            pairs, enriched, stats = build_pairs(rows, image_index, pred_root)
            print(f"    Matched: {stats['matched']}  Missed: {stats['missed']}")
            save_outputs(pairs, enriched, output_dir / book, prefix=book)

    print(f"\n✅ All done! Outputs in {output_dir}")


if __name__ == "__main__":
    main()
