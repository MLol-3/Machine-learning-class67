import numpy as np
import matplotlib.pyplot as plt
DEBUG = 0

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

# Gradient Descent function ใช้งานร่วมกันกับ cost function
def gradient_descent(x, y, initial_theta1, learning_rate, iterations):
    theta_vals = [initial_theta1]
    if(DEBUG):
        plot_line(x, initial_theta1)
    for _ in range(iterations):
        gradient = derivative_j_theta(x, y, theta_vals[-1])
        theta_new = theta_vals[-1] - learning_rate * gradient
        theta_vals.append(theta_new)
        if(DEBUG):
            plot_line(x, theta_new)
    return theta_vals

# function plot line 
def plot_line(x, theta1):
    Start_p = []
    Stop_p = []
    Min_p = min(x)
    Max_p = max(x)
    for i in range(Min_p-5, Max_p+5):
        Stop_p.append(i)
        Start_p.append(theta1 * i)
    plt.plot(Stop_p, Start_p , color = 'blue')

if __name__ == "__main__":
     # Data points
    x = np.array([0, 2])
    y = np.array([0, 2])
    plt.scatter(x, y, color = 'red', marker='X')
    learning_rate = 0.25
    iterations = 100
    initial_theta1 = 0
    theta_vals = gradient_descent(x, y, initial_theta1, learning_rate, iterations)
    print(f'theta1 = {theta_vals[-1]:.4f}')
    plot_line(x, theta_vals[-1])
    plt.xlabel('data X')
    plt.ylabel('data Y')
    plt.title(' Repersentation ')
    plt.legend(["Data points", "h(x) = theta * x"])
    plt.axis([-0.5, 3.5, -0.5, 3.5])
    plt.show()
