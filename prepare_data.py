import os
import cv2
import numpy as np
import albumentations as A

# Create Dataset Structure
DATASET_DIR = "dataset"
CLASSES = ["accident", "non_accident"]

for c in CLASSES:
    os.makedirs(os.path.join(DATASET_DIR, c), exist_ok=True)

# Augmentations for multi-angle and multi-weather simulation
transform_pipeline = A.Compose([
    # Angles & Transforms
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.7),
    A.Perspective(scale=(0.05, 0.1), p=0.5),
    
    # Weather & Lighting Variations (Summer, Winter/Snow, Rain simulation)
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.1, p=0.7),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
    A.MotionBlur(blur_limit=(3, 7), p=0.3),
    A.Resize(224, 224)
])

def generate_synthetic_placeholder_data():
    """Generates synthetic samples if local images are not present."""
    print("Generating baseline folder structure and placeholder images...")
    for label in CLASSES:
        target_dir = os.path.join(DATASET_DIR, label)
        # Create 200 samples per class if empty
        if len(os.listdir(target_dir)) == 0:
            for i in range(200):
                # Dummy synthetic images representing vehicle/object features
                img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
                aug_img = transform_pipeline(image=img)["image"]
                cv2.imwrite(os.path.join(target_dir, f"{label}_{i+1}.jpg"), aug_img)
    print("Data directory prepared successfully.")

if __name__ == "__main__":
    generate_synthetic_placeholder_data()
