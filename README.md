# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the required libraries and load the dataset for logistic regression.
2.Split the dataset into training and testing data, then normalize the input features.
3.Initialize the weights and bias values for the logistic regression model.
4.Apply Gradient Descent to update the weights and bias by minimizing the logistic loss function.
5.Predict the output classes using the trained model and evaluate the accuracy of the model.
``
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: S Rithika
RegisterNumber:  212225040344
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("Placement_Data (1).csv")

data1 = data.copy()

data1 = data1.drop(['sl_no', 'salary'], axis=1)

le = LabelEncoder()

data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])

x = data1.iloc[:, :-1].values
y = data1["status"].values

theta = np.random.randn(x.shape[1])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def loss(theta, x, y):
    h = sigmoid(x.dot(theta))
    return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

def gradient_descent(theta, x, y, alpha, num_iterations):

    m = len(y)

    for i in range(num_iterations):

        h = sigmoid(x.dot(theta))

        gradient = x.T.dot(h - y) / m

        theta = theta - alpha * gradient

    return theta

theta = gradient_descent(theta, x, y, alpha=0.01, num_iterations=1000)

def predict(theta, x):

    h = sigmoid(x.dot(theta))

    y_pred = np.where(h >= 0.5, 1, 0)

    return y_pred

y_pred = predict(theta, x)

accuracy = np.mean(y_pred.flatten() == y)

print("Accuracy:", accuracy)

print("Predicted:\n", y_pred)

print("Actual:\n", y)

xnew = np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])

y_prednew = predict(theta, xnew)

print("Predicted Result:", y_prednew)

*/
```

## Output:

<img width="938" height="398" alt="image" src="https://github.com/user-attachments/assets/b8a5b19d-7100-4c9d-a85b-06c94cfb970c" />

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

