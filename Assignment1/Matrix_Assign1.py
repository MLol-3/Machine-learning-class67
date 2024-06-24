import numpy as np
import matplotlib.pyplot as plt
DEBUG = 0
# Gradient Descent function
def gradient_descent(X, y, learning_rate=0.1, n_iterations=100):
    m = len(y)
    cost = []
    range_i = [i for i in range(n_iterations)]
    theta = np.zeros((X.shape[1]))  # initialization of theta
    for i in range(n_iterations):
        error = X.dot(theta) - y
        cost.append(np.mean(1-error))
        gradients = X.T.dot(error)
        theta = theta - learning_rate / m * gradients
    plt.plot(range_i, cost)
    return theta

if __name__ == "__main__":
    # Data x
    X = np.array([[2104, 1416, 1534, 852],
                [5, 3, 3, 2],
                [1, 2, 2, 1],
                [45, 40, 30, 36]])
    # Data y
    y = np.array([460, 232, 315, 178])
  
    X = (X - X.mean(axis=0)) / X.std(axis=0)    # Standardization
    X_b = np.c_[np.ones((X.shape[0], 1)), X]    # Add bias term
    # Using Gradient Descent
    theta_gd = gradient_descent(X_b, y)
    print("Parameters from Gradient Descent (theta):")
    print(theta_gd)

    # Predict using the estimated parameters
    X_new = np.array([[2104, 1416, 1534, 852],
                    [5, 3, 3, 2],
                    [1, 2, 2, 1],
                    [45, 40, 30, 36]])
    X_new = (X_new - X_new.mean(axis=0)) / X_new.std(axis=0) 
    X_new_b = np.c_[np.ones((len(X_new), 1)), X_new]  # Add bias term
    y_predict_gd = X_new_b.dot(theta_gd)

    print("Predictions using Gradient Descent:")
    print(y_predict_gd)
    plt.show()



