import matplotlib.pyplot as plt
import numpy as np

# Cost function
# calculates the average error of a linear regression model
#
# Needed To evaluate how well the model fits the data.
# ใช้เพือประเมินว่า model นั้นมีความเข้ากันกับ Data มากแค่ไหน
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
# derivative (slope) tells which direction to step to reduce the cost
#
# Needed To provide direction for minimizing the cost.
# ใช้เพื่อบอกว่าเราต้องเดินไปทางไหนถึงจะไปยังจุดที่ error น้อยที่สุด
def derivative_j_theta(x, y, theta1):
    n = len(y)
    error = []
    for i in range(n):
        hypothesis = theta1 * x[i]
        xi_error = (hypothesis - y[i]) * x[i]
        error.append(xi_error)
    dj_theta1 = (1/n) * sum(error)
    return dj_theta1


# Gradient Descent function ใช้งานร่วมกันกับ Derivative of the cost function

# theta_new = theta_old - learning_rate * (1/n) * sum((h(x_i) - y_i) * x_i)
# theta_new : Updated weight
# theta_old : Current weight
# learning_rate : Learning rate
# n : Number of data points
# h(x_i) : Hypothesis (predicted value) for data point x_i using the current weight
# y_i : Target label for data point x_i
# x_i : row of the input data matrix X 

# Needed for iteratively update the parameter and converge to an optimal value.
# ใช้เพื่อการ Update ค่า ตัวแปล theta และเพื่อหาค่าที่เบนเข้าหาจุดที่ดีที่สุด
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
    i = np.linspace(-4.5, 6.5, 100)
    J = j_theta(x, y, i)
    
    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle('Gradient Descent with Different Learning Rates', fontsize=16)

    # Applying gradient descent with different learning rates
    learning_rates = [0.1, 0.5, 0.75, 1.006]
    iterations = 8
    initial_theta1 = 6
    
    for idx, lr in enumerate(learning_rates): # 0, 0.1 > 1, 0.5 > 2, 0.75 > 3, 1.006

        # print(list(enumerate(learning_rates)))
        
        ax = axs[idx // 2, idx % 2] # Plot Block by Block Started with 0,0 > 0,1 > 1,0 > 1,1
        
        ax.plot(i, J, label='Cost Function') # Plot Cost Function (Parabola Graph)
        ax.set_xlabel("Weight 1")
        ax.set_ylabel("E")
        ax.set_title(f'Learning Rate: {lr}')
        
        theta_vals = gradient_descent(x, y, initial_theta1, lr, iterations)
        ax.plot(theta_vals, [j_theta(x, y, theta) for theta in theta_vals], '-o', label=f'Learning rate={lr}')
        
        ax.legend()
        # ax.grid(True)

    # plt.tight_layout()
    plt.show()
