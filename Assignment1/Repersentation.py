import numpy as np
import matplotlib.pyplot as plt

def j_theta(x, y, theta1):
    n = len(y)
    sq_error = []
    for i in range(n):
        hypothesis = theta1 * x[i]
        xi_error = (hypothesis - y[i])**2
        sq_error.append(xi_error)
    j_theta1 = (1 / (2 * n)) * sum(sq_error)
    return j_theta1

def Gradient_Descent(x, y, theta1, rate, alpha):
    plot_line(x, theta1)

    for _ in range(rate):
        theta1 = theta1 - alpha * 2 * (theta1-1)
        plot_line(x, theta1)
        mse = j_theta(x, y, theta1)
        if mse == 0:
            print(f"theta1 = {theta1}")
            break
    print(f"mse = {mse}")

def plot_line(x, theta1):
    p = []
    x_p = []
    Min = min(x)
    Max = max(x)
    for i in range(Min-5, Max+5):
        x_p.append(i)
        p.append(theta1 * i)
    plt.plot(x_p, p)

if __name__ == "__main__":
    x = np.array([0, 2])
    y = np.array([0, 2])
    plt.scatter(x, y)
    theta1 = 0
    Gradient_Descent(x, y, theta1, 100, 0.5)
    plt.axis([-0.5, 3.5, -0.5, 3.5])
    plt.show()
