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

# Derivative of the cost function 
def derivative_j_theta(x, y, theta0, theta1):
    n = len(y)
    error = []
    for i in range(n):
        hypothesis = theta0 + theta1 * x[i]
        xi_error = (hypothesis - y[i]) * x[i]
        error.append(xi_error)
    dj_theta1 = (1/n) * sum(error)
    return dj_theta1

# Gradient Descent function ใช้งานร่วมกันกับ cost function
def gradient_descent(X, y, learning_rate, iterations, initial_theta0=0, initial_theta1=10):
    n = len(X)
    theta0 = [initial_theta0]
    theta1 = [initial_theta1]
    for _ in range(iterations):
        # Derivative theta0 and theta1
        d_theta0 = []
        d_theta1 = []
        for i in range(n):
            d_theta0.append((theta0[-1] + theta1[-1] * X[i]) - y[i])
            d_theta1.append(((theta0[-1] + theta1[-1] * X[i]) - y[i]) * X[i])
        theta0_new = theta0[-1] - learning_rate * (1/n) * sum(d_theta0)
        theta1_new = theta1[-1] - learning_rate * (1/n) * sum(d_theta1)
        theta0.append(theta0_new)
        theta1.append(theta1_new)
    return theta0, theta1

if __name__ == "__main__":
    # Data points
    x = np.array([0, 2])
    y = np.array([0, 2])

    # สร้างเส้นในการเทียบ error_sam and theta_sam
    theta_sam  = np.linspace(-8,8)
    mse_sam = j_theta(x, y, 0, theta_sam)
    plt.plot(theta_sam, mse_sam)

    # init gradient descent value 
    learning_rate = 0.1
    iterations = 100
    initial_theta1 = -10
    theta0, theta1 = gradient_descent(x, y, learning_rate, iterations, initial_theta1=initial_theta1)

    # สร้างจุดที่ปรับค่า theta ใน function gradient_descent
    arr_theta1 = np.array(theta1)
    mse_theta1 = j_theta(x, y, 0, arr_theta1)
    plt.scatter(theta1, mse_theta1, color = 'red', marker='X') 

    plt.xlabel("theta1")
    plt.ylabel("E ")
    plt.title('Evaluation')
    plt.axis([-6, 8, -0.5, 40])
    plt.show()
    