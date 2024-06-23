import matplotlib.pyplot as plt
import numpy as np

# Cost Function
# calculates the average error of a linear regression model
def j_theta(x, y, theta1):
    n = len(y)
    sq_error = []
    for i in range(n):
        hypothesis = theta1 * x[i]
        xi_error = (hypothesis - y[i])**2
        sq_error.append(xi_error)
    j_theta1 = (1/(2*n)) * sum(sq_error)
    return j_theta1


# Added the Derivative_j_theta Function
# calculates how much to adjust the weight (theta1)
# to improve the fit of a straight line to the data.
def derivative_j_theta(x, y, theta1):
    n = len(y)
    error = []
    for i in range(n):
        hypothesis = theta1 * x[i]
        xi_error = (hypothesis - y[i]) * x[i]
        error.append(xi_error)
    dj_theta1 = (1/n) * sum(error)
    return dj_theta1
#End of Added the Derivative_j_theta Function


# Added the gradient_descent Function
###
### The goal of gradient descent is to find the values of theta 
### (including theta1) that minimize the cost function, 
### leading to the best possible fit for linear regression model
###
### we use gradient descent to find the optimal value of theta1
###
### By adjusting theta1 through gradient descent, 
### we minimize the overall error between the data and the model's predictions.
###

def gradient_descent(x, y, initial_theta1, learning_rate, iterations):
    theta_vals = [initial_theta1]
    for _ in range(iterations):
        gradient = derivative_j_theta(x, y, theta_vals[-1])
        theta_new = theta_vals[-1] - learning_rate * gradient
        theta_vals.append(theta_new)
    return theta_vals
# End of Added the gradient_descent Function

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
learning_rates = [0.1, 0.5, 0.75, 1.006]
i = np.linspace(-4.5, 6.5, 100)
x = [0, 2]
y = [0, 2]
J = j_theta(x, y, i)

# Added Line To Initiate Learning rate step, number of loop and start point of learning
learning_rates = [0.1, 0.5, 0.75, 1.006]
iterations = 10
initial_theta1 = 6
# End of Added Line To Initiate Learning rate step, number of loop and start point of learning

for idx, lr in enumerate(learning_rates):
    ax = axs[idx // 2, idx % 2]
    ax.plot(i, J)

    # Added Line To plot the different learning rates line
    theta_vals = gradient_descent(x, y, initial_theta1, lr, iterations) # Calculate to visual Learning line 
    ax.plot(theta_vals, [j_theta(x, y, theta) for theta in theta_vals], '-o', label=f'Learning rate={lr}') # Plot the Calculate to visual Learning line 
    # End of Added Line To plot the different learning rates line

    print(type(ax))
plt.show()