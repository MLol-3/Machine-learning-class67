import numpy as np
import matplotlib.pyplot as plt

# Cost function
def j_theta(X, y, theta0, theta1):
    n = len(y)
    sq_error = []
    for i in range(n):
        hypothesis = theta0 + theta1 * X[i]
        xi_error = (hypothesis - y[i])**2
        sq_error.append(xi_error)
    j_theta1 = (1 / (2 * n)) * sum(sq_error)
    return j_theta1

# function plot contour
def plot_theta(x, y,start, stop, num_contours=60):
    feature_x = np.linspace(start, stop, 100) 
    feature_y = np.linspace(start, stop, 100)
    [X, Y] = np.meshgrid(feature_x, feature_y)
    error = np.zeros_like(X)  
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            theta0 = X[i, j]  
            theta1 = Y[i, j]
            error[i, j] = j_theta(x, y, theta0, theta1)
    plt.contour(X, Y, error, num_contours, alpha=1.0, cmap='jet')  # Plot contour
    plt.xlabel('theta0')
    plt.ylabel('theta1')
    plt.title('Cost Function (Mean Squared Error) Contour')


if __name__ == "__main__":
    # Data points
    x = np.array([0, 2])
    y = np.array([0, 2])
    
	# init plot contour value 
    start = -8 
    stop = 8
    plot_theta(x, y, start, stop)
    plt.show()