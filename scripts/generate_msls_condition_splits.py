"""
Generate condition-specific MSLS-val splits for evaluating robustness
to illumination (night) and seasonal changes.

Outputs:
  - msls_val_night_qImages.npy   / msls_val_night_gt_25m.npy
  - msls_val_season_qImages.npy  / msls_val_season_gt_25m.npy

All splits share the same database images (msls_val_dbImages.npy).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cdist

MSLS_VAL_DIR = Path("datasets/msls-val")
CITIES = ["cph", "sf"]
DISTANCE_THRESHOLD = 25  # meters


def load_city_data(city: str):
    """Load postprocessed metadata and subtask index for a city."""
    base = MSLS_VAL_DIR / city
    q_meta = pd.read_csv(base / "query" / "postprocessed.csv")
    db_meta = pd.read_csv(base / "database" / "postprocessed.csv")
    q_subtask = pd.read_csv(base / "query" / "subtask_index.csv")
    return q_meta, db_meta, q_subtask


def compute_ground_truth(q_meta, db_meta, db_image_paths, q_keys, threshold=25):
    """
    Compute ground truth: for each query, find all DB images within `threshold` meters.
    Uses UTM easting/northing from postprocessed.csv.
    
    Returns list of arrays (one per query), each containing indices into db_image_paths.
    """
    # Build a key -> (easting, northing) map for DB
    db_key_to_idx = {}
    for i, path in enumerate(db_image_paths):
        key = Path(path).stem
        db_key_to_idx[key] = i

    # DB coordinates array (aligned with db_image_paths)
    db_coords = np.zeros((len(db_image_paths), 2))
    for _, row in db_meta.iterrows():
        key = row["key"]
        if key in db_key_to_idx:
            idx = db_key_to_idx[key]
            db_coords[idx] = [row["easting"], row["northing"]]

    ground_truth = []
    for q_key in q_keys:
        q_row = q_meta[q_meta["key"] == q_key]
        if q_row.empty:
            ground_truth.append(np.array([], dtype=np.int64))
            continue
        q_east = q_row["easting"].values[0]
        q_north = q_row["northing"].values[0]
        dists = np.sqrt((db_coords[:, 0] - q_east) ** 2 + (db_coords[:, 1] - q_north) ** 2)
        matches = np.where(dists < threshold)[0]
        ground_truth.append(matches)

    return ground_truth


def generate_split(condition_name: str, query_filter_fn):
    """
    Generate a condition-specific split.
    
    Args:
        condition_name: e.g. "night", "season"
        query_filter_fn: function(q_meta, q_subtask) -> list of query keys
    """
    # Load the shared database images
    db_images = np.load(MSLS_VAL_DIR / "msls_val_dbImages.npy")

    all_q_keys = []
    all_q_paths = []
    all_q_meta = []
    all_db_meta = []

    for city in CITIES:
        q_meta, db_meta, q_subtask = load_city_data(city)
        all_db_meta.append(db_meta)
        all_q_meta.append(q_meta)

        # Get filtered query keys for this city
        filtered_keys = query_filter_fn(q_meta, q_subtask)
        for key in filtered_keys:
            all_q_keys.append(key)
            all_q_paths.append(f"{city}/query/images/{key}.jpg")

    if len(all_q_keys) == 0:
        print(f"  [WARNING] No queries found for condition '{condition_name}'")
        return

    # Merge metadata from both cities
    merged_q_meta = pd.concat(all_q_meta, ignore_index=True)
    merged_db_meta = pd.concat(all_db_meta, ignore_index=True)

    # Compute ground truth
    gt = compute_ground_truth(merged_q_meta, merged_db_meta, db_images, all_q_keys, DISTANCE_THRESHOLD)

    # Filter out queries with no ground truth matches
    valid_indices = [i for i, g in enumerate(gt) if len(g) > 0]
    filtered_q_paths = [all_q_paths[i] for i in valid_indices]
    filtered_gt = [gt[i] for i in valid_indices]

    q_paths_arr = np.array(filtered_q_paths)
    gt_arr = np.array(filtered_gt, dtype=object)

    # Save
    out_q = MSLS_VAL_DIR / f"msls_val_{condition_name}_qImages.npy"
    out_gt = MSLS_VAL_DIR / f"msls_val_{condition_name}_gt_25m.npy"
    np.save(out_q, q_paths_arr)
    np.save(out_gt, gt_arr)

    print(f"  [{condition_name}] Queries: {len(filtered_q_paths)} (filtered from {len(all_q_keys)}), "
          f"DB: {len(db_images)}")
    print(f"  Saved: {out_q}")
    print(f"  Saved: {out_gt}")


def night_filter(q_meta, q_subtask):
    """Select night queries (n2d: night query → day database)."""
    night_keys = q_subtask[q_subtask["n2d"] == True]["key"].tolist()
    return night_keys


def season_filter(q_meta, q_subtask):
    """Select seasonal change queries (w2s + s2w)."""
    mask = (q_subtask["w2s"] == True) | (q_subtask["s2w"] == True)
    return q_subtask[mask]["key"].tolist()


def main():
    print("Generating MSLS-val condition-specific splits...")
    print(f"Dataset path: {MSLS_VAL_DIR}")
    print(f"Distance threshold: {DISTANCE_THRESHOLD}m\n")

    print("=== Night (n2d: night query → day database) ===")
    generate_split("night", night_filter)

    print("\n=== Season (w2s + s2w: cross-season matching) ===")
    generate_split("season", season_filter)

    print("\nDone!")


if __name__ == "__main__":
    main()
