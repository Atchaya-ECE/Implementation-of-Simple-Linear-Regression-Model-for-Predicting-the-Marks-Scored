# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries such as Pandas, NumPy, Matplotlib, and Scikit-learn modules for data handling, visualization, and linear regression.
2.Load the dataset containing the students' study hours and marks, then separate it into input feature (Hours) and target variable (Marks).
3.Split the dataset into training and testing sets, create a Simple Linear Regression model, train it using the training data, and predict the marks for the testing data.
4.Evaluate the model using MAE, MSE, and RMSE, visualize the regression line using plots, and use the trained model to predict marks for new study-hour values.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: ATCHAYA .V 
RegisterNumber: 212224060031 
*/
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from google.colab import files
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Upload Dataset
uploaded = files.upload()

# Read Dataset
df = pd.read_csv("exp_2_dataset_student_scores.csv")

# Display Dataset
print("First 5 Rows:")
print(df.head())

print("\nLast 5 Rows:")
print(df.tail())

# Input and Output
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

# Split Dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.33, random_state=0
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples :", len(X_test))

# Train Model
model = LinearRegression()
model.fit(X_train, Y_train)

print("\nCoefficient:", model.coef_[0])
print("Intercept :", model.intercept_)

# Predict
Y_pred = model.predict(X_test)

print("\nActual Values:")
print(Y_test)

print("\nPredicted Values:")
print(np.round(Y_pred, 2))

# Training Graph
plt.figure(figsize=(6,4))
plt.scatter(X_train, Y_train, color='orange', label='Training Data')
plt.plot(X_train, model.predict(X_train), color='red', label='Regression Line')
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours Studied")
plt.ylabel("Scores")
plt.legend()
plt.grid(True)
plt.show()

# Testing Graph
order = np.argsort(X_test.flatten())

plt.figure(figsize=(6,4))
plt.scatter(X_test, Y_test, color='blue', label='Testing Data')
plt.plot(
    X_test.flatten()[order],
    Y_pred[order],
    color='green',
    label='Prediction Line'
)
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours Studied")
plt.ylabel("Scores")
plt.legend()
plt.grid(True)
plt.show()

# Evaluation
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)

print("\nMean Absolute Error (MAE):", round(mae, 2))
print("Mean Squared Error (MSE):", round(mse, 2))
print("Root Mean Squared Error (RMSE):", round(rmse, 2))

# Predict New Values
new_hours = np.array([[2.5], [5.0], [8.0]])
predictions = model.predict(new_hours)

print("\nPredictions:")
for h, s in zip(new_hours.flatten(), predictions):
    print(f"Hours = {h} --> Predicted Score = {s:.2f}")

# Regression Equation
print("\nRegression Equation:")
print(f"Score = {model.coef_[0]:.4f} × Hours + {model.intercept_:.4f}")
```

## Output:
<img width="907" height="662" alt="image" src="https://github.com/user-attachments/assets/d00196bb-eebe-44b8-b227-da9fd43b991b" />

<img width="531" height="393" alt="image" src="https://github.com/user-attachments/assets/9f865aa4-4f15-4480-8aea-a80bf37d5e3b" />

<img width="656" height="267" alt="image" src="https://github.com/user-attachments/assets/98aa9b56-5200-4562-b202-a1db3a42a353" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
