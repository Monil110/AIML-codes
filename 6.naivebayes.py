import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class NaiveBayes:
    def __init__(self):
        self.class_prob = {}
        self.features_prob = {}

    def fit(self, x, y):
        for value in y:
            self.class_prob[value] = self.class_prob.get(value, 0) + 1

        counts = dict(self.class_prob)
        total = len(y)
        for key in self.class_prob:
            self.class_prob[key] /= total

        for c in self.class_prob:
            self.features_prob[c] = {}
            for feature in x.columns:
                self.features_prob[c][feature] = {}
                values = x[feature].unique()
                for value in values:
                    count = np.sum((x[feature] == value) & (y == c))
                    self.features_prob[c][feature][value] = count / counts[c]

    def predict(self, x):
        preds = []
        for i in range(len(x)):
            row = x.iloc[i]
            max_prob = -1
            pred_class = None
            for c in self.class_prob:
                prob = self.class_prob[c]
                for feature in x.columns:
                    value = row[feature]
                    prob *= self.features_prob[c][feature].get(value, 0)
                if prob > max_prob:
                    max_prob = prob
                    pred_class = c
            preds.append(pred_class)
        return preds

data = pd.read_csv("Social_Network_Ads.csv")
data["Gender"] = np.where(data["Gender"] == 'Male', 1, 0)
X = data.iloc[:, 1:4]
y = data['Purchased']
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = NaiveBayes()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

accuracy = accuracy_score(Y_pred, Y_test)
precision = precision_score(Y_pred, Y_test)
recall = recall_score(Y_pred, Y_test)
f1 = f1_score(Y_pred, Y_test)

print("Validation Set Metrics:")
print("Accuracy: {:.2f}".format(accuracy))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1 Score: {:.2f}".format(f1))

confusion = confusion_matrix(Y_pred, Y_test)
print(confusion)
print("Class 0 predicted and true:", confusion[0][0])
print("Class 0 predicted and false:", confusion[0][1])
print("Class 1 predicted and false:", confusion[1][0])
print("Class 1 predicted and true:", confusion[1][1])

valid = data.sample(n=20)
X_valid = valid.iloc[:, 1:4]
y_valid = valid['Purchased']
y_val = model.predict(X_valid)

accuracy = accuracy_score(y_val, y_valid)
precision = precision_score(y_val, y_valid)
recall = recall_score(y_val, y_valid)
f1 = f1_score(y_val, y_valid)

print("Validation Set Metrics:")
print("Accuracy: {:.2f}".format(accuracy))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1 Score: {:.2f}".format(f1))

confusion = confusion_matrix(y_val, y_valid)
print(confusion)
print("Class 0 predicted and true:", confusion[0][0])
print("Class 0 predicted and false:", confusion[0][1])
print("Class 1 predicted and false:", confusion[1][0])
print("Class 1 predicted and true:", confusion[1][1])

a = pd.DataFrame({"Gender": [1], "Age": [1], "EstimatedSalary": [0]})
print(model.predict(a))
