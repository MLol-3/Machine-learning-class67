import numpy as np
import matplotlib.pyplot as plt

def MSE(h, y):
    error = 0
    for i in range(len(h)):
        error += (h[i] - y[i])**2
    mse = 1/(2*2) * error
    return mse

def linear(w, x):
    f_x = []
    for i in x :
        f_x.append(w*i)
    return f_x

def Gradient_Descent(x, y, w, rate, a):
    f_x = linear(w, x)
    plot_line(x, w)

    for _ in range(rate):
        w = w - a * 2 * (w-1)
        plot_line(x, w)
        f_x = linear(w, x)
        mse = MSE(f_x, y)
        if mse == 0:
            print(f"w = {w}")
            break
    print(f"mse = {mse}")
    # return w

def plot_line(x, w):
    p = []
    x_p = []
    Min = min(x)
    Max = max(x)
    for i in range(Min-5, Max+5):
        x_p.append(i)
        p.append(w * i)
    plt.plot(x_p, p)


if __name__ == "__main__":
    x = np.array([0, 2])
    y = np.array([0, 2])
    plt.scatter(x, y)
    w = 0
    Gradient_Descent(x, y, w, 100, 0.5)
    plt.axis([-0.5, 3.5, -0.5, 3.5])
    plt.show()