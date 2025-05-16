import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class LogisticRegression:
    def __init__(self, learning_rate=0.01, iteration=1000):
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.theta = None

    def add_intercept(self, X):
        return np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = self.add_intercept(X)
        self.theta = np.zeros(X.shape[1])
        for _ in range(self.iteration):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.learning_rate * gradient

    def predict_prob(self, X):
        X = self.add_intercept(X)
        return self.sigmoid(np.dot(X, self.theta))

    def predict(self, X, threshold=0.5):
        return self.predict_prob(X) >= threshold

data = pd.read_csv("Breastcancer_data.csv")
X = data.iloc[:, 2:-1].values.astype(np.float64)
y = np.where(data.iloc[:, 1].values == 'M', 1, 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("Validation Set Metrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

conf = confusion_matrix(y_test, predictions)
print(conf)
print("Class 0 predicted and true :", conf[0][0])
print("Class 0 predicted and false :", conf[0][1])
print("Class 1 predicted and false :", conf[1][0])
print("Class 1 predicted and true :", conf[1][1])

import random
X_valid = [X[random.randint(0, 500)] for _ in range(20)]
Y_valid = [y[random.randint(0, 500)] for _ in range(20)]
print(Y_valid)
