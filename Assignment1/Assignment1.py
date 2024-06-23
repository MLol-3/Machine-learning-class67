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
def plot_contour(x, y,start, stop, num_contours=60):
    sam_theta0 = np.linspace(start, stop, 100) 
    sam_theta1 = np.linspace(start, stop, 100)
    [X, Y] = np.meshgrid(sam_theta0, sam_theta1)
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

# Gradient Descent function ใช้งานร่วมกันกับ cost function
# สูตร Wi = Wi - alpha * (1/n) * sigma((aX + b) - y) * X
# theta0 = b
# theta1 = a
# learning_rate = alpha

def gradient_descent(X, y, learning_rate, iterations, theta0 = 0, theta1 = 10):
    n = len(X)
    for _ in range(iterations):
        # Derivative theta0 and theta1
        d_theta0 = []
        d_theta1 = []
        for i in range(n):
            d_theta0.append((theta0+theta1*X[i])-y[i])
            d_theta1.append(((theta0+theta1*X[i])-y[i])*X[i])
        theta0 = theta0 - learning_rate*(1/n)*sum(d_theta0)
        theta1 = theta1 - learning_rate*(1/n)*sum(d_theta1)
        # plot theta0 and theta1
        plt.scatter(theta0, theta1, color = 'red', marker='X')  
    return theta0, theta1


if __name__ == "__main__":
    # Data points
    x = np.array([0, 2])
    y = np.array([0, 2])

    # init plot contour value 
    start = -8 
    stop = 8
    plot_contour(x, y, start, stop)

    # init gradient descent value 
    learning_rate = 0.5
    iterations = 1000
    theta0, theta1 = gradient_descent(x, y, learning_rate, iterations)
    print(f"theta0 =  {theta0:.4f}\ntheta1 = {theta1:.4f} ")
    # set magin 
    plt.axis([-8, 8, -8, 8])
    plt.show()