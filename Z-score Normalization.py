import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# === Input and Output Paths ===
input_folder = ''  # 🔁 Change this
output_folder = '' # 🔁 Change this

os.makedirs(output_folder, exist_ok=True)

# === Function: Z-Score Normalization (per RGB channel) ===
def z_score_normalize_color(image):
    norm_image = np.zeros_like(image, dtype=np.float32)
    for c in range(3):  # R, G, B channels
        channel = image[:, :, c]
        mean = np.mean(channel)
        std = np.std(channel)
        if std == 0:
            norm_image[:, :, c] = channel
        else:
            norm_image[:, :, c] = (channel - mean) / std
    return norm_image

# === Process Images ===
for filename in tqdm(os.listdir(input_folder)):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        img_path = os.path.join(input_folder, filename)

        # Load image in color (BGR format)
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            continue

        # Convert to RGB for display/normalization
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Z-Score normalization on RGB channels
        norm_rgb = z_score_normalize_color(image_rgb)

        # Rescale normalized image to [0, 255] and convert to uint8
        norm_rescaled = cv2.normalize(norm_rgb, None, 0, 255, cv2.NORM_MINMAX)
        norm_rescaled = norm_rescaled.astype(np.uint8)

        # Convert back to BGR for saving
        norm_bgr = cv2.cvtColor(norm_rescaled, cv2.COLOR_RGB2BGR)

        # Save normalized image
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, norm_bgr)

        # Display input and output images side-by-side
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(image_rgb)
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(norm_rescaled)
        plt.title("Z-Score Normalized")
        plt.axis('off')

        plt.suptitle(filename)
        plt.tight_layout()
        plt.show()