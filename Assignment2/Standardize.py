# Import numpy
import numpy as np

# Given data x
x = np.array([[2104, 1416, 1534, 852],
              [5, 3, 3, 2],
              [1, 2, 2, 1],
              [45, 40, 30, 36]])

# Standardization of x
x_std = (x - np.mean(x, axis=0)) / np.std(x, axis=0)

# Given data y
y = np.array([460, 232, 315, 178])


print("Standardized x:")
print(x_std)

