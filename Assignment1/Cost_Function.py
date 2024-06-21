import matplotlib.pyplot as plt
import numpy as np


def j_theta(x, y, theta1):
    n = len(y)
    sq_error = []
    for i in range(n):
        hypothensis = theta1 * x[i]
        xi_error = (hypothensis - y[i])**2
        sq_error.append(xi_error)
    j_theta1 = (1/(2*n)) * sum(sq_error)
    return j_theta1

if __name__ == "__main__":
    x = [0, 2]
    y = [0, 2]
    
    for i in range(-5,8):
        theta_out = j_theta(x, y, i)
        plt.scatter(i, theta_out)

    E  = np.linspace(-5,8)
    w = j_theta(x, y, E)
    print(w)
    plt.plot(E, w)
    plt.xlabel("w1")
    plt.ylabel("E ")
    plt.axis([-6, 8, -0.5, 40])
    plt.show()
    