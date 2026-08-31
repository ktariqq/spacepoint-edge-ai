"""
Builds the tinyML training set from the TU Graz Aerial Semantic
Segmentation Drone Dataset. This replaces the earlier general-photo +
K-means pseudo-labeling approach: this dataset gives real, human-
annotated ground truth masks, and its images are actual nadir drone
photos at the altitude the payload will use.

Splits by SOURCE IMAGE (not by tile) into train/val before cropping -
tiles from the same photo look almost identical, so splitting after
tiling would let near-duplicate tiles leak across the train/val
boundary and inflate validation accuracy.

Output structure: common/data/tinyml_dataset/{train,val}/<category>/*.png
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# Adjust these two to match your actual unzipped folder layout - different
# Kaggle mirrors of this dataset lay it out slightly differently.
ORIGINAL_IMAGES_DIR = Path("data/raw_tugraz/dataset/semantic_drone_dataset/original_images")
LABEL_MASKS_DIR = Path("data/raw_tugraz/dataset/semantic_drone_dataset/label_images_semantic")
CLASS_DICT_CSV = Path("data/raw_tugraz/class_dict_seg.csv")

OUTPUT_DIR = Path("data/tinyml_dataset")
TILE_SIZE = 256
FINAL_SIZE = 48
DARK_BRIGHTNESS_THRESHOLD = 60
VAL_FRACTION = 0.2
RANDOM_SEED = 42

CATEGORIES = ["vegetation", "bright_surface", "shadow", "bare_soil", "water", "asphalt", "other"]

# Maps this dataset's 24 original classes onto the project's 7 categories.
# "shadow" isn't a native class here - it's derived separately below from
# tile brightness, the same rule the project's own image pipeline uses.
CLASS_REMAP = {
    "unlabeled": "other",
    "paved-area": "asphalt",
    "dirt": "bare_soil",
    "grass": "vegetation",
    "gravel": "bare_soil",
    "water": "water",
    "rocks": "bare_soil",
    "pool": "water",
    "vegetation": "vegetation",
    "roof": "bright_surface",
    "wall": "bright_surface",
    "window": "bright_surface",
    "door": "bright_surface",
    "fence": "other",
    "fence-pole": "other",
    "person": "other",
    "dog": "other",
    "car": "other",
    "bicycle": "other",
    "tree": "vegetation",
    "bald-tree": "other",
    "ar-marker": "other",
    "obstacle": "other",
    "conflicting": None,  # dropped, not a real category
}


def load_class_index_to_category():
    """Reads class_dict_seg.csv from the download and builds a lookup
    from mask pixel value to project category, using the file's own
    row order rather than an assumed order - mirrors of this dataset
    have occasionally reordered rows."""
    class_table = pd.read_csv(CLASS_DICT_CSV)
    return {
        index: CLASS_REMAP.get(row["name"].strip(), "other")
        for index, row in class_table.iterrows()
    }


def main():
    index_to_category = load_class_index_to_category()
    for split in ["train", "val"]:
        for category in CATEGORIES:
            (OUTPUT_DIR / split / category).mkdir(parents=True, exist_ok=True)

    image_paths = sorted(ORIGINAL_IMAGES_DIR.glob("*.jpg"))
    print(f"Found {len(image_paths)} source images")

    # Split by whole source image, not by tile
    rng = random.Random(RANDOM_SEED)
    shuffled = image_paths.copy()
    rng.shuffle(shuffled)
    val_count = int(len(shuffled) * VAL_FRACTION)
    val_image_names = {p.name for p in shuffled[:val_count]}

    tile_count = 0
    for image_path in image_paths:
        mask_path = LABEL_MASKS_DIR / (image_path.stem + ".png")
        if not mask_path.exists():
            print(f"No mask for {image_path.name}, skipping")
            continue

        split = "val" if image_path.name in val_image_names else "train"

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        height, width, _ = image.shape

        for y in range(0, height - TILE_SIZE, TILE_SIZE):
            for x in range(0, width - TILE_SIZE, TILE_SIZE):
                image_tile = image[y:y+TILE_SIZE, x:x+TILE_SIZE]
                mask_tile = mask[y:y+TILE_SIZE, x:x+TILE_SIZE]

                category_counts = {c: 0 for c in CATEGORIES}
                for class_index, count in zip(*np.unique(mask_tile, return_counts=True)):
                    category = index_to_category.get(int(class_index))
                    if category is not None:
                        category_counts[category] += int(count)

                majority_category = max(category_counts, key=category_counts.get)

                mean_brightness = image_tile.mean()
                if mean_brightness < DARK_BRIGHTNESS_THRESHOLD and majority_category != "water":
                    majority_category = "shadow"

                small_tile = Image.fromarray(image_tile).resize((FINAL_SIZE, FINAL_SIZE))
                out_path = OUTPUT_DIR / split / majority_category / f"{tile_count:05d}.png"
                small_tile.save(out_path)
                tile_count += 1

        print(f"Processed {image_path.name} ({split}), running total: {tile_count}")

    print(f"\nDone. {tile_count} tiles written.")
    for split in ["train", "val"]:
        print(f"{split}:")
        for category in CATEGORIES:
            count = len(list((OUTPUT_DIR / split / category).glob("*.png")))
            print(f"  {category}: {count}")


if __name__ == "__main__":
    main()