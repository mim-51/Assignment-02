import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

# Define the function f(x, y)
def f(x, y):
    return y**2 - 2 * y * np.cos(x)

# Create x and y values within the specified range
x = np.linspace(1, 7, 100)  # x ranges from 1 to 7
y = np.linspace(-5, 5, 100) # y values can range from -5 to 5 for better visualization

# Create meshgrid for x and y
X, Y = np.meshgrid(x, y)

# Compute Z values based on the function
Z = f(X, Y)

# Plotting the 3D surface
fig = plt.figure( )
ax = plt.axes( projection='3d')

# Plot surface
ax.plot_surface(X, Y, Z, cmap='viridis')

# Adding labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f(x, y)')
ax.set_title('3D Plot of $f(x,y)=y^2-2y\cos(x)$')

# Show the plot
plt.show()



