"""
Converts a few sample images into normalized float32 C arrays so the
firmware has something to classify without a camera attached yet.

IMPORTANT: MicroTFLite's ModelSetInput() takes the REAL (unquantized)
input value and quantizes it internally using the model's own
scale/zero-point, since the input tensor is int8. It must NOT be
handed an already-quantized int8 value - that quantizes twice and
produces garbage. So this script outputs the same kind of value the
model was trained and calibrated on: pixel intensity normalized to
[0, 1], as float32.
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
        pixels = (np.array(image).astype(np.float32) / 255.0).flatten()
        out.write(f"const float {var_name}[{len(pixels)}] = {{\n")
        for i in range(0, len(pixels), 8):
            chunk = pixels[i:i + 8]
            out.write("  " + ", ".join(f"{v:.6f}f" for v in chunk) + ",\n")
        out.write("};\n\n")
    out.write("#endif\n")

print("Wrote esp32/models/test_images.h (float32, normalized 0-1)")