import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# === Paths ===
input_folder = ''      # Change this
output_folder = ''    # Change this
os.makedirs(output_folder, exist_ok=True)

# === Augmentation Functions ===

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

def scale_image(image, scale=1.2):
    (h, w) = image.shape[:2]
    scaled = cv2.resize(image, (int(w * scale), int(h * scale)))
    return cv2.resize(scaled, (w, h))  # Resize back to original

def flip_image(image, mode):
    # 1: horizontal, 0: vertical
    return cv2.flip(image, mode)

# === Display Function ===

def display_images(title, images, titles):
    plt.figure(figsize=(20, 6))
    for i, (img, t) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i+1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(t)
        plt.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# === Process Images ===

for filename in tqdm(os.listdir(input_folder)):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        img_path = os.path.join(input_folder, filename)
        image = cv2.imread(img_path)
        if image is None:
            continue

        base_name, ext = os.path.splitext(filename)

        # Augmentations
        rotated_45 = rotate_image(image, 45)
        rotated_90 = rotate_image(image, 90)
        rotated_180 = rotate_image(image, 180)
        scaled = scale_image(image, 1.2)
        flipped_h = flip_image(image, 1)
        flipped_v = flip_image(image, 0)

        # Display
        display_images(
            title=f"Augmentations: {filename}",
            images=[image, rotated_45, rotated_90, rotated_180, scaled, flipped_h, flipped_v],
            titles=['Original', 'Rotated 45°', 'Rotated 90°', 'Rotated 180°', 'Scaled', 'Flipped H', 'Flipped V']
        )

        # Save (Optional)
        cv2.imwrite(os.path.join(output_folder, f"{base_name}_rot45{ext}"), rotated_45)
        cv2.imwrite(os.path.join(output_folder, f"{base_name}_rot90{ext}"), rotated_90)
        cv2.imwrite(os.path.join(output_folder, f"{base_name}_rot180{ext}"), rotated_180)
        cv2.imwrite(os.path.join(output_folder, f"{base_name}_scaled{ext}"), scaled)
        cv2.imwrite(os.path.join(output_folder, f"{base_name}_flipH{ext}"), flipped_h)
        cv2.imwrite(os.path.join(output_folder, f"{base_name}_flipV{ext}"), flipped_v)