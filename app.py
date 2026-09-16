import os
import numpy as np
from PIL import Image, ImageDraw

def create_synthetic_dataset(base_path="dataset", num_per_class=200):
    classes = ["accident", "non_accident"]
    
    for cls in classes:
        folder = os.path.join(base_path, cls)
        os.makedirs(folder, exist_ok=True)
        
        for i in range(1, num_per_class + 1):
            # Create a 224x224 RGB image with random environmental color variations
            bg_color = (
                np.random.randint(100, 200),
                np.random.randint(100, 200),
                np.random.randint(100, 200)
            )
            img = Image.new("RGB", (224, 224), color=bg_color)
            draw = ImageDraw.Draw(img)
            
            if cls == "accident":
                # Draw overlapping shapes representing colliding vehicles / impact zone
                draw.rectangle([40, 40, 140, 140], fill=(200, 30, 30), outline=(0, 0, 0))
                draw.rectangle([100, 100, 190, 190], fill=(50, 50, 50), outline=(255, 255, 0))
                draw.line([(40, 40), (190, 190)], fill=(255, 0, 0), width=4)
            else:
                # Draw separated shapes representing non-colliding vehicles in traffic
                draw.rectangle([20, 20, 90, 90], fill=(30, 100, 200), outline=(0, 0, 0))
                draw.rectangle([130, 130, 200, 200], fill=(50, 50, 50), outline=(255, 255, 255))
            
            img.save(os.path.join(folder, f"{cls}_{i:03d}.jpg"))

    print(f"Dataset successfully created with 200 'accident' and 200 'non_accident' images.")

if __name__ == "__main__":
    create_synthetic_dataset()
