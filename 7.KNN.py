import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, x, y):
        self.x = x
        self.y = y

    def predict_single(self, point):
        distances = [euclidean_distance(point, data_point) for data_point in self.x]
        sorted_indices = np.argsort(distances)
        k_nearest_labels = [self.y[i] for i in sorted_indices[:self.k]]
        return np.argmax(np.bincount(k_nearest_labels))

    def predict(self, x):
        return [self.predict_single(point) for point in x]

df = pd.read_csv('/Users/swastikagarwal/Downloads/SEM5_RVCE/LAB WORK/AIML/iris_csv (1).csv')

x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

label_map = {'Iris-versicolor': 1, 'Iris-virginica': 2, 'Iris-setosa': 3}
y = np.array([label_map[label] for label in y])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

model = KNN()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("accuracy:", accuracy_score(y_test, y_pred))
print("precision:", precision_score(y_test, y_pred, average='macro'))
print("recall:", recall_score(y_test, y_pred, average='macro'))
print("f1 score:", f1_score(y_test, y_pred, average='macro'))

valid = df.sample(n=20)
x_valid = valid.iloc[:, :-1].values
y_valid = np.array([label_map[label] for label in valid.iloc[:, -1]])

y_predict = model.predict(x_valid)

print("validation accuracy:", accuracy_score(y_valid, y_predict))
