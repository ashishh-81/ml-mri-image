import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.exceptions import ConvergenceWarning
import warnings

# ------------------------------
# Load Images
# ------------------------------
def load_images_from_folder(root_folder, img_size=(128, 128)):
    X, y, class_names = [], [], []
    for split in os.listdir(root_folder):
        split_path = os.path.join(root_folder, split)
        if not os.path.isdir(split_path):
            continue
        for class_name in os.listdir(split_path):
            folder = os.path.join(split_path, class_name)
            if not os.path.isdir(folder):
                continue
            if class_name not in class_names:
                class_names.append(class_name)
            label = class_names.index(class_name)
            for file in os.listdir(folder):
                if not file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    continue
                try:
                    img = load_img(os.path.join(folder, file), target_size=img_size)
                    X.append(img_to_array(img))
                    y.append(label)
                except Exception as e:
                    print(f"⚠️ Skipping {file}: {e}")
    X, y = np.array(X), np.array(y)
    print(f"✅ Loaded {len(X)} images from {len(class_names)} classes: {class_names}")
    return X, y, class_names

# ------------------------------
# Feature Extraction using VGG16
# ------------------------------
def extract_features_vgg16(X):
    print("🔍 Extracting features using VGG16...")
    vgg = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    X_prep = preprocess_input(X)
    features = vgg.predict(X_prep, verbose=1)
    features = features.reshape(features.shape[0], -1)
    print(f"✅ Extracted feature shape: {features.shape}")
    return features

# ------------------------------
# Clustering + Visualization
# ------------------------------
def evaluate_clustering(X, y_true, n_clusters=4):
    results = {}
    print("⚙️ Reducing dimensions using PCA (50 components)...")
    pca_50 = PCA(n_components=50, random_state=42)
    X_reduced = pca_50.fit_transform(X)
    X_pca = PCA(n_components=2, random_state=42).fit_transform(X_reduced)

    # --- KMeans ---
    print("🔹 Running K-Means...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    y_kmeans = kmeans.fit_predict(X_reduced)
    results['KMeans'] = (silhouette_score(X_reduced, y_kmeans),
                         adjusted_rand_score(y_true, y_kmeans))

    # --- Gaussian Mixture ---
    print("🔹 Running Gaussian Mixture Model...")
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            gmm = GaussianMixture(n_components=n_clusters, reg_covar=1e-3, random_state=42)
            y_gmm = gmm.fit_predict(X_reduced)
            results['GMM'] = (silhouette_score(X_reduced, y_gmm),
                              adjusted_rand_score(y_true, y_gmm))
    except Exception as e:
        print(f"⚠️ GMM failed: {e}")
        y_gmm = np.zeros_like(y_true)
        results['GMM'] = (0, 0)

    # --- Hierarchical ---
    print("🔹 Running Hierarchical Clustering...")
    hier = AgglomerativeClustering(n_clusters=n_clusters)
    y_hier = hier.fit_predict(X_reduced)
    results['Hierarchical'] = (silhouette_score(X_reduced, y_hier),
                               adjusted_rand_score(y_true, y_hier))

    # --- Plot ---
    plt.figure(figsize=(15, 5))
    titles = ['K-Means', 'GMM', 'Hierarchical']
    for i, (labels, title) in enumerate(zip([y_kmeans, y_gmm, y_hier], titles)):
        plt.subplot(1, 3, i + 1)
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='plasma', s=8)
        plt.title(title)
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
    plt.suptitle("🧠 MRI Clustering Results (2D PCA Projection)")
    plt.tight_layout()
    plt.show(block=True)

    print("\n📊 Clustering Evaluation (Silhouette | ARI):")
    for model, (sil, ari) in results.items():
        print(f"{model:15s} → Silhouette: {sil:.4f} | ARI: {ari:.4f}")

    return results

# ------------------------------
# MAIN
# ------------------------------
if __name__ == "__main__":
    root_folder = "dataset"  # Change this if needed
    X, y, class_names = load_images_from_folder(root_folder)
    if len(X) == 0:
        print("❌ No images found. Check your dataset path.")
        exit()
    features = extract_features_vgg16(X)
    results = evaluate_clustering(features, y, n_clusters=len(class_names))
