import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

df = pd.read_csv(r"C:\Users\MONIL\Desktop\Codes\AIML codes\iris_csv (1).csv")
data = df.iloc[:, :4].values

kmeans = KMeansCustom(n_clusters=3)
centroids = kmeans.initialize_centroids(data)

for _ in range(kmeans.max_iterations):
    clusters = kmeans.assign_to_clusters(data, centroids)
    new_centroids = kmeans.update_centroids(data, clusters)
    if np.all(centroids == new_centroids):
        break
    centroids = new_centroids

plt.scatter(data[:, 0], data[:, 1], c=clusters)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x')
plt.title('KMeans Clustering')
plt.show()
