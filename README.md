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
data = pd.read_csv("Placement_Data (1).csv")
print(data.head())
print(data.info())
print(data.isnull().sum())
print(data["status"].value_counts())
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
categorical_cols = [
    "gender",
    "ssc_b",
    "hsc_b",
    "hsc_s",
    "degree_t",
    "workex",
    "specialisation",
    "status"
]

for col in categorical_cols:
    data[col] = le.fit_transform(data[col])
data["salary"] = data["salary"].fillna(data["salary"].mean())

print(data.head())
x = data[[
    "ssc_p",
    "hsc_p",
    "degree_p",
    "etest_p",
    "mba_p",
    "salary"
]]

y = data["status"]
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=100
)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
sample_data = pd.DataFrame(
    [[70, 75, 80, 85, 78, 300000]],
    columns=[
        "ssc_p",
        "hsc_p",
        "degree_p",
        "etest_p",
        "mba_p",
        "salary"
    ]
)

prediction = dt.predict(sample_data)

print("Prediction:", prediction)

if prediction[0] == 1:
    print("Placed")
else:
    print("Not Placed")
*/
```

## Output:

<img width="919" height="735" alt="image" src="https://github.com/user-attachments/assets/9c96b9a8-9721-4292-a911-8b1d8d1ed816" />
<img width="848" height="754" alt="image" src="https://github.com/user-attachments/assets/80597ab3-b083-47f4-8fd4-ce05c7ec3e57" />

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

