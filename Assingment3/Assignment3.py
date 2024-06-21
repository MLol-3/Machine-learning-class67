import numpy as np
import matplotlib.pyplot as plt

# Function to minimize (example: f(x) = x^2)
def f(x):
    return x**2

# Derivative of the function (gradient)
def df(x):
    return 2*x

# Gradient Descent function
def gradient_descent(x_init, learning_rate, iterations):
    x_vals = [x_init]
    for _ in range(iterations):
        gradient = df(x_vals[-1])
        x_new = x_vals[-1] - learning_rate * gradient
        x_vals.append(x_new)
    return x_vals


if __name__ == "__main__":
    # Create x values for plotting the function
    x = np.linspace(-5, 5, 100)
    y = f(x)

    # Initialize plots for each learning rate scenario
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    learning_rates = [0.1, 0.5, 0.75, 1.01]
    iterations = 100
    x_init = 4.5

    for i, lr in enumerate(learning_rates):
        ax = axs[i // 2, i % 2]
        ax.plot(x, y, label='f(x) = x^2')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-1, 35)
        ax.set_title(f'Learning Rate: {lr}')
        
        x_vals = gradient_descent(x_init, lr, iterations)
        ax.plot(x_vals, [f(x) for x in x_vals], 'r', label='Gradient Descent')
        ax.plot(x_vals[-1], f(x_vals[-1]), 'ro')
        
        ax.legend()

    plt.suptitle('Gradient Descent Visualization with Different Learning Rates', fontsize=16)
    plt.tight_layout()
    plt.show()
