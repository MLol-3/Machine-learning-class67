import numpy as np
DEBUG = 0
# Gradient Descent function
def gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    m = len(y)
    theta = np.zeros((X_b.shape[1]))  # initialization of theta
    for _ in range(n_iterations):
        gradients = X.T.dot(X.dot(theta) - y)
        theta = theta - learning_rate / m * gradients
    if (DEBUG):
        print(f'X.dot(theta) = {X.dot(theta)}\n')
        print(f'y = {y}\n')
        print(f'X.dot(thea) - y = {X.dot(theta) - y}\n')
        print(f'X.T = {X.T}\n')
        print(f'X.T.dot(X.dot(theta) - y) = {gradients}\n')
        print(f'theta = {theta}\n')
    return theta

if __name__ == "__main__":
    # Given data x
    X = np.array([0, 2])
    # Given data y
    y = np.array([0, 2])
  
    # X = (X - X.mean(axis=0)) / X.std(axis=0)    # Normalization 
    X_b = np.c_[np.ones((X.shape[0], 1)), X]    # Add bias term
    # print(X_b)

    # Using Gradient Descent
    theta_gd = gradient_descent(X_b, y)
    print("Parameters from Gradient Descent (theta):")
    print(theta_gd)

    # Predict using the estimated parameters
    X_new = np.array([3, 6])
    X_new_b = np.c_[np.ones((len(X_new), 1)), X_new]  # Add bias term
    y_predict_gd = X_new_b.dot(theta_gd)

    print("Predictions using Gradient Descent:")
    print(y_predict_gd)



