import matplotlib.pyplot as plt
import numpy as np


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

# Gradient Descent function ใช้งานร่วมกันกับ cost function
def gradient_descent(x, y, initial_theta1, learning_rate, iterations):
    theta_vals = [initial_theta1]
    for _ in range(iterations):
        # plot theta เทียบ J_theta ในค่าที่ปรับจริง
        E = j_theta(x, y, 0, theta_vals[-1])
        plt.scatter(theta_vals[-1], E, color = 'red', marker='X') 

        gradient = derivative_j_theta(x, y, theta_vals[-1])
        theta_new = theta_vals[-1] - learning_rate * gradient
        theta_vals.append(theta_new)
    return theta_vals

if __name__ == "__main__":
    # Data points
    x = np.array([0, 2])
    y = np.array([0, 2])

    # สร้างเส้นในการเทียบ E and theta1 
    theta1 = np.linspace(-8,8)
    E = j_theta(x, y, 0, theta1)
    plt.plot(theta1, E)

    # init gradient descent value 
    learning_rate = 0.1
    iterations = 100
    initial_theta1 = 10
    gradient_descent(x, y, initial_theta1, learning_rate, iterations)

    plt.xlabel("theta1")
    plt.ylabel("E ")
    plt.title('Evaluation')
    plt.axis([-6, 8, -0.5, 40])
    plt.show()
    