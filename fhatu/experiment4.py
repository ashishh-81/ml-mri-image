
import os
import sys
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

# Image I/O and preprocessing
try:
    from skimage.io import imread
    from skimage.color import rgb2gray
    from skimage.transform import resize as sk_resize
except Exception as e:
    # Fallback to matplotlib if scikit-image isn't available
    import matplotlib.image as mpimg
    def imread(p):
        return mpimg.imread(p)
    def rgb2gray(img):
        if img.ndim == 3:
            return img.mean(axis=2)
        return img
    def sk_resize(img, out_shape):
        # simple, not ideal fallback (preserves range)
        from PIL import Image
        if img.ndim == 2:
            mode = 'F'
        else:
            mode = 'RGB'
        im = Image.fromarray((img*255).astype(np.uint8)) if img.max()<=1.0 else Image.fromarray(img.astype(np.uint8))
        im = im.convert('L') if mode!='RGB' else im.convert('RGB')
        im = im.resize(out_shape[::-1])  # PIL expects (W,H)
        return np.array(im)/255.0

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample
from sklearn.base import clone

# -----------------------------
# Config
# -----------------------------
TRAIN_DIR = "archive/Training"
TEST_DIR  = "archive/Testing"
IMG_SIZE  = (64, 64)          # H x W
N_COMPONENTS = 100            # PCA components (tune as needed)
RANDOM_STATE = 42
N_BOOTSTRAPS = 20             # number of bootstrap samples for Bayesian approx

np.random.seed(RANDOM_STATE)



def list_images_with_labels(root_dir):
    """Return (paths, labels, class_names) from directory of class subfolders."""
    class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    label_map = {cls: i for i, cls in enumerate(class_names)}

    paths, labels = [], []
    for cls in class_names:
        cls_dir = os.path.join(root_dir, cls)
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
            for p in glob.glob(os.path.join(cls_dir, ext)):
                paths.append(p)
                labels.append(label_map[cls])
    return np.array(paths), np.array(labels), class_names


def load_and_preprocess(paths, labels, img_size=IMG_SIZE):
    """Load images, preprocess, and keep labels consistent."""
    X, y_new, kept_paths = [], [], []
    for p, lbl in tqdm(zip(paths, labels), total=len(paths), desc="Loading images", ncols=80):
        try:
            img = imread(p)
            if img.dtype != np.float32:
                img = img.astype(np.float32)
            if img.max() > 1.0:
                img = img / 255.0
            if img.ndim == 3:
                img = rgb2gray(img)
            img = sk_resize(img, img_size)
            X.append(img.reshape(-1))
            y_new.append(lbl)
            kept_paths.append(p)
        except Exception as e:
            print(f"Warning: could not read {p} ({e})", file=sys.stderr)
            continue
    return np.array(X, dtype=np.float32), np.array(y_new), np.array(kept_paths)


# -----------------------------
# Load data
# -----------------------------
print("\n[1/6] Scanning dataset...")
train_paths, y_train, class_names = list_images_with_labels(TRAIN_DIR)
test_paths,  y_test,  _           = list_images_with_labels(TEST_DIR)
print(f"Classes: {class_names}")
print(f"Train images: {len(train_paths)} | Test images: {len(test_paths)}")

print("\n[2/6] Loading & preprocessing images...")
X_train_img, y_train, train_paths = load_and_preprocess(train_paths, y_train, IMG_SIZE)
X_test_img,  y_test,  test_paths  = load_and_preprocess(test_paths,  y_test,  IMG_SIZE)

print(f"Train: {X_train_img.shape}, Labels: {y_train.shape}")
print(f"Test: {X_test_img.shape}, Labels: {y_test.shape}")


# -----------------------------
# Build shared feature pipeline (Scaler + PCA)
# -----------------------------
print("\n[3/6] Fitting feature pipeline (StandardScaler + PCA)...")
feature_pipe = Pipeline([
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("pca", PCA(n_components=min(N_COMPONENTS, X_train_img.shape[1]), random_state=RANDOM_STATE))
])

X_train = feature_pipe.fit_transform(X_train_img)
X_test  = feature_pipe.transform(X_test_img)
print(f"Feature shape after PCA: train {X_train.shape} | test {X_test.shape}")


print("\n[4/6] Training Bayesian Logistic Regression via bootstrap...")
base_logreg = LogisticRegression(
    multi_class='auto', solver='lbfgs', max_iter=1000, C=1.0, random_state=RANDOM_STATE
)

proba_stack = []
for b in range(N_BOOTSTRAPS):
    # sample with replacement from training indices
    idx = resample(np.arange(len(X_train)), replace=True, random_state=RANDOM_STATE + b)
    Xb, yb = X_train[idx], y_train[idx]
    clf = clone(base_logreg)
    clf.fit(Xb, yb)
    proba_stack.append(clf.predict_proba(X_test))

proba_stack = np.stack(proba_stack, axis=0)        # (B, n_test, n_classes)
proba_mean = proba_stack.mean(axis=0)              # posterior predictive mean
proba_std  = proba_stack.std(axis=0)               # predictive uncertainty per class

y_pred_bayes = proba_mean.argmax(axis=1)
acc_bayes = accuracy_score(y_test, y_pred_bayes)
print(f"Bayesian-LogReg Accuracy: {acc_bayes:.4f}")
print("\nBayesian-LogReg Classification Report:")
print(classification_report(y_test, y_pred_bayes, target_names=class_names))
print("Confusion Matrix (Bayesian-LogReg):")
print(confusion_matrix(y_test, y_pred_bayes))

# Save predictions
bayes_out = pd.DataFrame({
    "path": test_paths,
    "true_label": [class_names[i] for i in y_test],
    "pred_label": [class_names[i] for i in y_pred_bayes]
})
# Add mean prob and std for the predicted class
pred_class_indices = y_pred_bayes
pred_mean = proba_mean[np.arange(len(y_pred_bayes)), pred_class_indices]
pred_std  = proba_std[np.arange(len(y_pred_bayes)), pred_class_indices]
bayes_out["pred_prob_mean"] = pred_mean
bayes_out["pred_prob_std"]  = pred_std
bayes_out.to_csv("bayesian_logreg_predictions.csv", index=False)
print("Saved: bayesian_logreg_predictions.csv")

print("\n[5/6] Training SVM (RBF kernel)...")
svm_clf = SVC(kernel='rbf', C=10.0, gamma='scale', probability=False, random_state=RANDOM_STATE)
svm_clf.fit(X_train, y_train)

y_pred_svm = svm_clf.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {acc_svm:.4f}")
print("\nSVM Classification Report:")
print(classification_report(y_test, y_pred_svm, target_names=class_names))
print("Confusion Matrix (SVM):")
print(confusion_matrix(y_test, y_pred_svm))

svm_out = pd.DataFrame({
    "path": test_paths,
    "true_label": [class_names[i] for i in y_test],
    "pred_label": [class_names[i] for i in y_pred_svm]
})
svm_out.to_csv("svm_predictions.csv", index=False)
print("Saved: svm_predictions.csv")

# -----------------------------
# Wrap up
# -----------------------------
print("\n[6/6] Done.")
print(f"Classes: {class_names}")
print(f"Bayesian-LogReg Accuracy: {acc_bayes:.4f} | SVM Accuracy: {acc_svm:.4f}")
print("Outputs: bayesian_logreg_predictions.csv, svm_predictions.csv")
