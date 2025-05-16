import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from google.colab import drive

class KMeansCustom:
    def __init__(self, n_clusters, max_iterations=100):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations

    def initialize_centroids(self, data):
        random_indices = np.random.choice(data.shape[0], self.n_clusters, replace=False)
        centroids = data[random_indices]
        return centroids

    def assign_to_clusters(self, data, centroids):
        clusters = []
        for point in data:
            distances = [np.linalg.norm(point - centroid) for centroid in centroids]
            closest = np.argmin(distances)
            clusters.append(closest)
        return np.array(clusters)

    def update_centroids(self, data, clusters):
        new_centroids = []
        for i in range(self.n_clusters):
            points = data[clusters == i]
            if len(points) > 0:
                centroid = points.mean(axis=0)
            else:
                centroid = np.zeros(data.shape[1])
            new_centroids.append(centroid)
        return np.array(new_centroids)

    def fit(self, data):
        centroids = self.initialize_centroids(data)
        for _ in range(self.max_iterations):
            clusters = self.assign_to_clusters(data, centroids)
            new_centroids = self.update_centroids(data, clusters)
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
        self.cen = centroids
        self.lab = clusters

# Mount Google Drive to access dataset
drive.mount('/content/drive')

iris = pd.read_csv('/content/drive/MyDrive/Datasets/iris_csv (1).csv')
data = iris.iloc[:, 2:4].values  # Using petal length and petal width

k = 4
model = KMeansCustom(n_clusters=k)
model.fit(data)

plt.scatter(data[:, 0], data[:, 1], c=model.lab, cmap='viridis')
plt.scatter(model.cen[:, 0], model.cen[:, 1], marker='X', s=200, c='red')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Custom K-means Clustering on Iris Dataset')
plt.show()
