"""
Converts a few sample images into int8 C arrays so the firmware has
something to classify without a camera attached yet.
"""

from pathlib import Path
import numpy as np
from PIL import Image

IMAGE_SIZE = 48
SAMPLE_IMAGES = [
    ("common/data/tinyml_dataset/val/vegetation/00691.png", "test_image_vegetation"),
    ("common/data/tinyml_dataset/val/bright_surface/00873.png", "test_image_bright"),
]

with open("esp32/models/test_images.h", "w") as out:
    out.write("#ifndef TEST_IMAGES_H\n#define TEST_IMAGES_H\n\n")
    for path, var_name in SAMPLE_IMAGES:
        image = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        pixels = np.array(image).astype(np.int32) - 128  # shift to int8 range
        flat = pixels.astype(np.int8).flatten()
        out.write(f"const int8_t {var_name}[{len(flat)}] = {{\n")
        for i in range(0, len(flat), 12):
            out.write("  " + ", ".join(str(v) for v in flat[i:i+12]) + ",\n")
        out.write("};\n\n")
    out.write("#endif\n")

print("Wrote esp32/models/test_images.h")