import os
import cv2
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte


# ===== 3.4.1 GEOMETRIC FEATURES =====

# a. Convex Hull Feature
def extract_convex_hull(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)
        return len(hull)  # number of convex hull points
    return 0

# b. Skeletonization Feature
def extract_skeleton(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    binary = binary // 255
    skeleton = skeletonize(binary).astype(np.uint8)
    return int(np.sum(skeleton))  # number of skeleton pixels


# ===== 3.4.2 MOTION TRACKING =====

# a. Optical Flow using Farneback
def compute_optical_flow(prev_img, next_img):
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mean_mag = np.mean(mag)
    mean_ang = np.mean(ang)
    return mean_mag, mean_ang  # motion features

# b. Simple Trajectory Estimation (Mean Flow Vector)
def compute_trajectory(flow):
    return np.mean(flow, axis=(0, 1)).tolist()


# ===== 3.4.3 TEXTURE & COLOR HISTOGRAM =====

# a. Color Histogram (HSV 3D histogram)
def extract_color_histogram(image, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                        [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist.tolist()

# b. GLCM Texture Features
def compute_glcm(image):
    max_gray = 256
    glcm = np.zeros((max_gray, max_gray), dtype=np.float32)
    for i in range(image.shape[0] - 1):
        for j in range(image.shape[1] - 1):
            row = image[i, j]
            col = image[i, j + 1]
            glcm[row, col] += 1
    glcm /= (np.sum(glcm) + 1e-10)
    return glcm

def extract_glcm_features(image):
    image = cv2.resize(image, (128, 128))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = compute_glcm(gray)

    contrast = np.sum([(i - j)**2 * glcm[i, j] for i in range(256) for j in range(256)])
    energy = np.sum(glcm ** 2)
    entropy = -np.sum(glcm * np.log2(glcm + 1e-10))
    homogeneity = np.sum([glcm[i, j] / (1 + (i - j)**2) for i in range(256) for j in range(256)])
    dissimilarity = np.sum([np.abs(i - j) * glcm[i, j] for i in range(256) for j in range(256)])

    mean_i = np.sum([i * np.sum(glcm[i, :]) for i in range(256)])
    mean_j = np.sum([j * np.sum(glcm[:, j]) for j in range(256)])
    std_i = np.sqrt(np.sum([(i - mean_i)**2 * np.sum(glcm[i, :]) for i in range(256)]))
    std_j = np.sqrt(np.sum([(j - mean_j)**2 * np.sum(glcm[:, j]) for j in range(256)]))

    correlation = np.sum([((i - mean_i)*(j - mean_j)*glcm[i, j])
                         for i in range(256) for j in range(256)]) / (std_i * std_j + 1e-10)

    return [contrast, correlation, energy, entropy, homogeneity, dissimilarity]


# ===== FEATURE EXTRACTION WRAPPER FOR FOLDER =====

def extract_all_features_from_folder(folder_path, output_csv):
    files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    results = []

    for i in range(len(files)-1):  # for motion tracking, use consecutive pairs
        f1 = os.path.join(folder_path, files[i])
        f2 = os.path.join(folder_path, files[i+1])

        img1 = cv2.imread(f1)
        img2 = cv2.imread(f2)
        if img1 is None or img2 is None:
            continue

        convex_feat = extract_convex_hull(img1)
        skeleton_feat = extract_skeleton(img1)
        mag, ang = compute_optical_flow(img1, img2)
        color_hist = extract_color_histogram(img1)
        glcm_feat = extract_glcm_features(img1)

        # Combine all features
        feature_vector = [files[i], convex_feat, skeleton_feat, mag, ang] + color_hist + glcm_feat
        results.append(feature_vector)

    # Column headers
    color_cols = [f'hist_bin_{i}' for i in range(512)]  # 8x8x8 = 512 bins
    glcm_cols = ['contrast', 'correlation', 'energy', 'entropy', 'homogeneity', 'dissimilarity']
    columns = ['filename', 'convex_hull_pts', 'skeleton_pixels', 'mean_flow_mag', 'mean_flow_angle'] + color_cols + glcm_cols

    # Save to CSV
    df = pd.DataFrame(results, columns=columns)

    # Add random binary labels (0 or 1)
    df['label'] = np.random.randint(0, 2, size=len(df))

    # Save final DataFrame with labels
    df.to_csv(output_csv, index=False)
    print(f"Feature extraction with labels completed. Results saved to: {output_csv}")
    return df


# ===== RUN EXAMPLE =====

# Update these paths
input_folder = ''  # Set your folder with images
output_csv = ''

# Run extraction
df = extract_all_features_from_folder(input_folder, output_csv)

# Display first few rows
df.head()
