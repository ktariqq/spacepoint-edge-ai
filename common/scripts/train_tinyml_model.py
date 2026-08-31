"""
Trains a small CNN on the auto-labeled dataset from Step 3, small
enough to fit on the ESP32 once quantized.
"""

import tensorflow as tf

DATA_DIR = "data/tinyml_dataset"
IMAGE_SIZE = 48
BATCH_SIZE = 32
EPOCHS = 20

train_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/train",
    image_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE,
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/val",
    image_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names
print("Classes:", class_names)

normalize = tf.keras.layers.Rescaling(1.0 / 255)
train_ds = train_ds.map(lambda x, y: (normalize(x), y))
val_ds = val_ds.map(lambda x, y: (normalize(x), y))

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    tf.keras.layers.Conv2D(8, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(16, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(len(class_names), activation="softmax"),
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

model.save("models/land_cover_model.keras")
print("Saved to models/land_cover_model.keras")
print("Class order (remember this for the firmware):", class_names)