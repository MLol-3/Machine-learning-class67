import numpy as np
import matplotlib.pyplot as plt
DEBUG = 0
# Normal Equation
def normal_equation(X, y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

# Gradient Descent function
def gradient_descent(X, y, learning_rate=0.5, n_iterations=1000):
    m = len(y)
    theta = np.zeros((X.shape[1]))  # initialization of theta
    for _ in range(n_iterations):
        gradients = X.T.dot(X.dot(theta) - y)
        theta = theta - learning_rate / m * gradients
    return theta

if __name__ == "__main__":
    # Data points
    X = np.array([0, 2])
    y = np.array([0, 2])

    # plt.scatter(X, y)
    # X = (X - X.mean(axis=0)) / X.std(axis=0)
    # Add x0 = 1 to each instance for the bias term
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
    # print(X_b)

    # Using Normal Equation
    theta_normal = normal_equation(X_b, y)
    print("Parameters from Normal Equation (theta):")
    print(theta_normal)

    # Using Gradient Descent
    theta_gd = gradient_descent(X_b, y)
    print("Parameters from Gradient Descent (theta):")
    print(theta_gd)

    # Predict using the estimated parameters
    X_new = np.array([3, 7])
    X_new_b = np.c_[np.ones((X.shape[0], 1)), X_new]  # Add bias term
    y_predict_normal = X_new_b.dot(theta_normal)
    y_predict_gd = X_new_b.dot(theta_gd)

    print(f"Predictions using Normal Equation:")
    print(y_predict_normal)
    print("Predictions using Gradient Descent:")
    print(y_predict_gd)

    plt.show()
