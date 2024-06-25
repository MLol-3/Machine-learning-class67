import numpy as np
import matplotlib.pyplot as plt
DEBUG = 0

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


# Derivative of the cost function with respect to theta1
def derivative_j_theta(x, y, theta1):
    n = len(y)
    error = []
    for i in range(n):
        hypothesis = theta1 * x[i]
        xi_error = (hypothesis - y[i]) * x[i]
        error.append(xi_error)
    dj_theta1 = (1/n) * sum(error)
    return dj_theta1

# Gradient Descent function for cost function line plotting 
def gradient_descent_line(x, y, initial_theta1, learning_rate, iterations):
    theta_vals = [initial_theta1]
    for _ in range(iterations):
        gradient = derivative_j_theta(x, y, theta_vals[-1])
        theta_new = theta_vals[-1] - learning_rate * gradient
        theta_vals.append(theta_new)
        if DEBUG:
            plot_line(x, theta_new)
    return theta_vals

# Plot line function
def plot_line(x, theta1):
    Start_p = []
    Stop_p = []
    Min_p = min(x)
    Max_p = max(x)
    for i in range(Min_p-5, Max_p+5):
        Stop_p.append(i)
        Start_p.append(theta1 * i)
    plt.plot(Stop_p, Start_p, color='blue')


# Plot contour function
def plot_contour(x, y, start, stop, num_contours=60):
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


# Gradient Descent function for contour plotting
# Gradient Descent function ใช้งานร่วมกันกับ cost function
# สูตร Wi = Wi - alpha * (1/n) * sigma((aX + b) - y) * X
# theta0 = b
# theta1 = a
# learning_rate = alpha
def gradient_descent(X, y, learning_rate, iterations, theta0=0, theta1=10):
    n = len(X)
    for _ in range(iterations):
        # Derivative theta0 and theta1
        d_theta0 = []
        d_theta1 = []
        for i in range(n):
            d_theta0.append((theta0 + theta1 * X[i]) - y[i])
            d_theta1.append(((theta0 + theta1 * X[i]) - y[i]) * X[i])
        theta0 = theta0 - learning_rate * (1/n) * sum(d_theta0)
        theta1 = theta1 - learning_rate * (1/n) * sum(d_theta1)
        # plot theta0 and theta1
        plt.scatter(theta0, theta1, color='red', marker='X')
    return theta0, theta1

if __name__ == "__main__":
    # Data points
    x = np.array([0, 2])
    y = np.array([0, 2])
    
    # Plotting the data points and gradient descent line iterations
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(x, y, color='red', marker='X')
    learning_rate = 0.25
    iterations = 100
    initial_theta1 = 0
    theta_vals = gradient_descent_line(x, y, initial_theta1, learning_rate, iterations)
    print(f'theta1 = {theta_vals[-1]:.4f}')
    plot_line(x, theta_vals[-1])
    plt.xlabel('data X')
    plt.ylabel('data Y')
    plt.title('Data Points and Line Iterations')
    plt.legend(["Data points", "h(x) = theta * x"])
    plt.axis([-0.5, 3.5, -0.5, 3.5])
    
    # Plotting the contour and gradient descent path
    plt.subplot(1, 2, 2)

    # init plot contour value 
    start = -8
    stop = 8
    plot_contour(x, y, start, stop)

    # init gradient descent value 
    learning_rate = 0.5
    iterations = 1000
    theta0, theta1 = gradient_descent(x, y, learning_rate, iterations)
    print(f"theta0 =  {theta0:.4f}\ntheta1 = {theta1:.4f}")
    
    # set magin 
    plt.axis([-8, 8, -8, 8])
    plt.show()