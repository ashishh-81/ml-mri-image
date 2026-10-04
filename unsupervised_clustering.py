import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ----- Feature extraction -----
def batch_features(img_paths, model, batch_size=32):
    feats = []
    for i in range(0, len(img_paths), batch_size):
        batch = []
        for path in img_paths[i:i+batch_size]:
            try:
                img = image.load_img(path, target_size=(224,224))
                arr = image.img_to_array(img)
                batch.append(arr)
            except:
                continue
        if batch:
            batch = preprocess_input(np.array(batch))
            feats.extend(model.predict(batch, verbose=0))
        print(f"Processed {min(i+batch_size,len(img_paths))}/{len(img_paths)} images")
    return np.array(feats)

def load_data(path='dataset'):
    img_paths, labels = [], []
    for root, _, files in os.walk(path):
        for f in files:
            if f.lower().endswith(('.jpg','.png')):
                img_paths.append(os.path.join(root,f))
                labels.append(os.path.basename(root))
    base = VGG16(weights='imagenet', include_top=False, pooling='avg')
    model = Model(base.input, base.output)
    X = batch_features(img_paths, model)
    return X, np.array(labels)

# ----- Clustering & visualization -----
def cluster_and_print(X, y, method, n_clusters=4):
    if method=='kmeans': labels = KMeans(n_clusters).fit_predict(X)
    elif method=='gmm': labels = GaussianMixture(n_clusters).fit_predict(X)
    else: labels = AgglomerativeClustering(n_clusters).fit_predict(X)
    print(f"{method.upper()} → Silhouette: {silhouette_score(X,labels):.4f} | ARI: {adjusted_rand_score(y,labels):.4f}")
    pca = PCA(2).fit_transform(X)
    plt.scatter(pca[:,0], pca[:,1], c=labels, cmap='rainbow', s=20)
    plt.title(f"{method.upper()} Clusters (PCA 2D)"); plt.show()
    if method=='hierarchical':
        dendrogram(linkage(X[:200], 'ward'), truncate_mode='level', p=5)
        plt.title("Hierarchical Dendrogram"); plt.show()

# ----- Main -----
X, y = load_data('dataset')
for m in ['kmeans','gmm','hierarchical']:
    cluster_and_print(X, y, m)
