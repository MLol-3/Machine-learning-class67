import matplotlib.pyplot as plt
import numpy as np


def j_theta(x, y, theta1):
    n = len(y)
    sq_error = []
    for i in range(n):
        hypothesis = theta1 * x[i]
        xi_error = (hypothesis - y[i])**2
        sq_error.append(xi_error)
    j_theta1 = (1/(2*n)) * sum(sq_error)
    return j_theta1


fig, axs = plt.subplots(2, 2, figsize=(10, 10))
learning_rates = [0.1, 0.5, 0.75, 1.006]
i = np.linspace(-4.5, 6.5, 100)
x = [0, 2]
y = [0, 2]
J = j_theta(x, y, i)
for idx, lr in enumerate(learning_rates):
        ax = axs[idx // 2, idx % 2]
        ax.plot(i, J, label='Cost Function')
        print(type(ax))
plt.show()