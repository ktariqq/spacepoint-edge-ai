"""
Converts the trained Keras model to a fully int8-quantized TFLite
model - this is what makes it small and fast enough for the ESP32.
"""

import numpy as np
import tensorflow as tf

IMAGE_SIZE = 48

model = tf.keras.models.load_model("common/models/land_cover_model.keras")

train_ds = tf.keras.utils.image_dataset_from_directory(
    "common/data/tinyml_dataset/train", image_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=1,
)

def representative_dataset():
    for images, _ in train_ds.take(200):
        yield [tf.cast(images, tf.float32) / 255.0]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open("esp32/models/land_cover_model.tflite", "wb") as f:
    f.write(tflite_model)

print(f"Model size: {len(tflite_model) / 1024:.1f} KB")