import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras import layers, models

DATASET_DIR = "dataset"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15

images, labels = [], []
label_map = {"non_accident": 0, "accident": 1}

# Load images
for category, label in label_map.items():
    folder = os.path.join(DATASET_DIR, category)
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(folder, filename)
            img = load_img(img_path, target_size=IMAGE_SIZE)
            images.append(img_to_array(img))
            labels.append(label)

X = preprocess_input(np.array(images, dtype="float32"))
y = np.array(labels)

# Dataset Split (70% Train, 15% Validation, 15% Test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

print(f"Train count: {len(X_train)} | Val count: {len(X_val)} | Test count: {len(X_test)}")

# Data Augmentation & Transfer Learning Architecture
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

inputs = layers.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Model Training
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# Test Set Metrics
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype("int32").flatten()

print("\n" + "="*40)
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score : {f1_score(y_test, y_pred):.4f}")
print("="*40)
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Non-Accident", "Accident"]))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Export Trained Weights
model.save("accident_model.h5")
print("\nModel saved successfully as 'accident_model.h5'")
