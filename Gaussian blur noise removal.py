import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# === Paths ===
input_folder = ''      # 🔁 Change this
output_folder = ''
os.makedirs(output_folder, exist_ok=True)

# === Gaussian Blur Function ===
def apply_gaussian_blur(image, kernel_size=(5, 5), sigma=1.0):
    """
    Apply Gaussian blur using OpenCV.
    - kernel_size: (width, height), should be odd.
    - sigma: Standard deviation (σ) of the Gaussian function.
    """
    return cv2.GaussianBlur(image, kernel_size, sigma)

# === Display Function ===
def display_images(original, blurred, title):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
    plt.title("Gaussian Blurred")
    plt.axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# === Process Images ===
for filename in tqdm(os.listdir(input_folder)):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        input_path = os.path.join(input_folder, filename)
        image = cv2.imread(input_path)

        if image is None:
            continue

        # Apply Gaussian Blur
        blurred = apply_gaussian_blur(image, kernel_size=(5, 5), sigma=1.0)

        # Save output
        output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_gaussian_blur.jpg")
        cv2.imwrite(output_path, blurred)

        # Display original and blurred
        display_images(image, blurred, title=filename)
