# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries and generate the input dataset for logistic regression.

2.Initialize the model parameters, learning rate, and number of iterations.

3.Apply the sigmoid function and train the model using gradient descent to update the parameter values.

4.Predict the output classes using the trained logistic regression model and calculate the accuracy.

5.Display the final parameter values and plot the data points with the decision boundary graph.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: DHARSHINI.M
RegisterNumber: 212225220025 
*/import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
X = np.random.randn(100, 2)
Y = (X[:, 0] + X[:, 1] > 0).astype(int)   
X = np.c_[np.ones(X.shape[0]), X]  
theta = np.zeros(X.shape[1])  
learning_rate = 0.1
epochs = 1000
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
for i in range(epochs):
    z = np.dot(X, theta)
    h = sigmoid(z)
    gradient = np.dot(X.T, (h - Y)) / Y.size
    theta -= learning_rate * gradient
print("Final Parameters (theta):", theta)
def predict(X_new):
    X_new = np.c_[np.ones(X_new.shape[0]), X_new]
    return (sigmoid(np.dot(X_new, theta)) >= 0.5).astype(int)
Y_pred = predict(X[:, 1:])
accuracy = np.mean(Y_pred == Y)
print("Accuracy:", accuracy)
plt.figure(figsize=(6, 5))
plt.scatter(X[:, 1], X[:, 2], c=Y, cmap='bwr', label='Actual')
x_values = [np.min(X[:, 1]), np.max(X[:, 1])]
y_values = -(theta[0] + np.dot(theta[1], x_values)) / theta[2]
plt.plot(x_values, y_values, color='black', label='Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('Logistic Regression using Gradient Descent')
plt.show()
```

## Output:
<img width="865" height="651" alt="image" src="https://github.com/user-attachments/assets/67253e87-e79a-477c-8e1e-62485351bfc4" />



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

