import numpy as np
import matplotlib.pyplot as plt

# Create a grid of points
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

# Define the function f(x, y) = 4x^2 + y^2
Z = 4*X*2 + Y*2

# Create the contour plot with specific levels k = 1, 4, 9, 16, 25, 36
levels = [1, 4, 9, 16, 25, 36]
plt.contour(X, Y, Z, levels, cmap='viridis')  # Apply color map

plt.colorbar()  # Show color bar
plt.title(r'Contour Plot of $f(x, y) = 4x^2 + y^2$')
plt.xlabel('X')
plt.ylabel('Y')

# Show the plot
plt.show()