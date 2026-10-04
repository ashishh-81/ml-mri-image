# --- Principal Component Analysis (PCA) Example ---
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Step 1: Load dataset (you can replace this with your own)
data = load_iris()
X = data.data
y = data.target
target_names = data.target_names

# Step 2: Standardize the data (important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply PCA (reduce to 2 components)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Step 4: Explained variance
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total variance captured by 2 components:", np.sum(pca.explained_variance_ratio_))

# Step 5: Plot the PCA results
plt.figure(figsize=(7,5))
for i, target in enumerate(np.unique(y)):
    plt.scatter(X_pca[y == target, 0], X_pca[y == target, 1], label=target_names[i], s=40)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - 2D Visualization of Iris Dataset")
plt.legend()
plt.grid(True)
plt.show()
