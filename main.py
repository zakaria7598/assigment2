import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv('ai4i2020.csv')

print(data.head())

data = data.drop(["UDI","Product ID","Type","TWF","HDF","PWF","OSF","RNF"], axis=1)

data1 = data.drop('Machine failure', axis=1)  
data2 = data['Machine failure']  

data1 = pd.get_dummies(data1)

data1 = data1.fillna(data1.mean())

X_train, X_test, y_train, y_test = train_test_split(data1, data2, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{conf_matrix}")

class_report = classification_report(y_test, y_pred)
print(f"Classification Report:\n{class_report}")

plt.scatter(range(len(y_pred)), y_pred)
plt.show()