import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate some sample data
np.random.seed(3)
X = 2 * np.random.rand(100, 1)
print(X)
y = 4 + 3 * X + np.random.randn(100, 1)
print(y)

# Function to compute the cost function
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/(2*m)) * np.sum(np.square(predictions - y))
    return cost

# Function to perform gradient descent
def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations, 2))
    
    for i in range(iterations):
        predictions = X.dot(theta)
        errors = np.dot(X.transpose(), (predictions - y))
        theta -= learning_rate * (1/m) * errors
        theta_history[i, :] = theta.transpose()
        cost_history[i] = compute_cost(X, y, theta)
        
    return theta, cost_history, theta_history

# Add intercept term to X
X_b = np.c_[np.ones((len(X), 1)), X]

# Initialize theta and hyperparameters
theta_initial = np.random.randn(2, 1)
learning_rate = 0.9
iterations = 10000

# Perform gradient descent
theta_final, cost_history, theta_history = gradient_descent(X_b, y, theta_initial, learning_rate, iterations)

# Plotting the cost function contours
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
theta0_mesh, theta1_mesh = np.meshgrid(theta0_vals, theta1_vals)
J_vals = np.zeros_like(theta0_mesh)

for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        theta = np.array([theta0_vals[i], theta1_vals[j]])
        J_vals[i, j] = compute_cost(X_b, y, theta)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(theta0_mesh, theta1_mesh, J_vals.transpose(), cmap='viridis')
ax.set_xlabel('Theta 0')
ax.set_ylabel('Theta 1')
ax.set_zlabel('Cost')
ax.set_title('Cost Function Contours')
ax.view_init(30, 120)
plt.show()

# Plotting the parameter adjustment steps
plt.figure(figsize=(12, 8))
plt.contour(theta0_mesh, theta1_mesh, J_vals.transpose(), np.logspace(-2, 3, 20), cmap='viridis')
plt.scatter(theta_history[:, 0], theta_history[:, 1], c='r')
plt.xlabel('Theta 0')
plt.ylabel('Theta 1')
plt.title('Gradient Descent: Minimizing Cost Function')
plt.colorbar(label='Cost')
plt.show()

# Printing the final parameters
print("Final theta parameters:", theta_final.ravel())
