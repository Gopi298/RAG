import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---------------- Config ----------------
DATA_DIR = "dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 25
SEED = 42
MODEL_PATH = "models/accident_model.keras"

os.makedirs("models", exist_ok=True)
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ---------------- Data Generators with Augmentation ----------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode="nearest",
    validation_split=0.2          # 20% for validation from the whole set
)

# No heavy augmentation for validation/test
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    seed=SEED,
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    seed=SEED,
    shuffle=False
)

print("Class indices:", train_gen.class_indices)   # {'accident': 0, 'non_accident': 1} or reverse

# ---------------- Model (Transfer Learning - EfficientNetB0) ----------------
base = applications.EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(*IMG_SIZE, 3)
)
base.trainable = False   # freeze first

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.4),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
             tf.keras.metrics.Recall(name="recall")]
)

# Callbacks
early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-6)
checkpoint = callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True)

# ---------------- Train ----------------
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=[early_stop, reduce_lr, checkpoint]
)

# Optional fine-tuning (unfreeze top layers)
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
             tf.keras.metrics.Recall(name="recall")]
)

history_fine = model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen,
    callbacks=[early_stop, reduce_lr, checkpoint]
)

# ---------------- Evaluation on validation set ----------------
val_gen.reset()
y_true = val_gen.classes
y_pred_prob = model.predict(val_gen).ravel()
y_pred = (y_pred_prob > 0.5).astype(int)

print("\n===== Classification Report =====")
print(classification_report(y_true, y_pred, target_names=list(train_gen.class_indices.keys())))

print("Accuracy :", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall   :", recall_score(y_true, y_pred))
print("F1-score :", f1_score(y_true, y_pred))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=list(train_gen.class_indices.keys()),
            yticklabels=list(train_gen.class_indices.keys()))
plt.title("Confusion Matrix")
plt.ylabel("True")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("models/confusion_matrix.png")
plt.show()

# Save final model (already saved by checkpoint, but ensure)
model.save(MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")
