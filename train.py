import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory as IMD

# Settings
batch_size = 32
img_height = 180
img_width = 180

# 1. Load Kaggle Dataset ('ckay16/accident-detection-from-cctv-footage')
train_ds = IMD(
    './data/train',
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='binary'
)

val_ds = IMD(
    './data/val',
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='binary'
)

# 2. Rescale pixel values [0, 1]
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# 3. Build CNN Architecture matching the notebook
model = models.Sequential([
    layers.Input(shape=(img_height, img_width, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary output: >0.5 -> Accident
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 4. Train Model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# 5. Save model
model.save("accident_model.h5")
print("Model trained and saved as accident_model.h5")
