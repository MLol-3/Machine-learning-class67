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

if __name__ == "__main__":
    # Data points
    x = np.array([0, 2])
    y = np.array([0, 2])

    # สร้างเส้นในการเทียบ E and theta1 
    theta_sam  = np.linspace(-8,8)
    mse_sam = j_theta(x, y, 0, theta_sam)
    plt.plot(theta_sam, mse_sam)
    plt.xlabel("theta1")
    plt.ylabel("E ")
    plt.title('Evaluation')
    plt.axis([-6, 8, -0.5, 40])
    plt.show()
    