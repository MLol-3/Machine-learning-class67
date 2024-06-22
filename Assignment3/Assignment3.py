import matplotlib.pyplot as plt
import numpy as np

# Cost function
def j_theta(x, y, theta1):
    n = len(y)
    sq_error = []
    for i in range(n):
        hypothesis = theta1 * x[i]
        xi_error = (hypothesis - y[i])**2
        sq_error.append(xi_error)
    j_theta1 = (1/(2*n)) * sum(sq_error)
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

# Gradient Descent function incorporating the cost function
def gradient_descent(x, y, initial_theta1, learning_rate, iterations):
    theta_vals = [initial_theta1]
    for _ in range(iterations):
        gradient = derivative_j_theta(x, y, theta_vals[-1])
        theta_new = theta_vals[-1] - learning_rate * gradient
        theta_vals.append(theta_new)
    return theta_vals

if __name__ == "__main__":
    x = [0, 2]
    y = [0, 2]
    
    # Plotting the original cost function
    i = np.linspace(-6, 8, 100)
    J = j_theta(x, y, i)
    
    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle('Gradient Descent with Different Learning Rates', fontsize=16)

    # Applying gradient descent with different learning rates
    learning_rates = [0.1, 0.5, 0.75, 1.01]
    iterations = 15
    initial_theta1 = 6
    
    for idx, lr in enumerate(learning_rates):
        ax = axs[idx // 2, idx % 2]
        
        ax.plot(i, J, label='Cost Function')
        ax.set_xlabel("Weight 1")
        ax.set_ylabel("E")
        ax.set_title(f'Learning Rate: {lr}')
        
        theta_vals = gradient_descent(x, y, initial_theta1, lr, iterations)
        ax.plot(theta_vals, [j_theta(x, y, theta) for theta in theta_vals], '-o', label=f'Learning rate={lr}')
        
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()
