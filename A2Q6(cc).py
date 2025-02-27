import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import dblquad
from mpl_toolkits.mplot3d import Axes3D

# Define variables
x, y = sp.symbols('x y')

# Paraboloid z = 4 - x^2 - y^2
z = 4 - x*2 - y*2

# Cylinder (x - 1)^2 + y^2 = 1 -> bounds for y
y_upper = sp.sqrt(1 - (x - 1)**2)
y_lower = -sp.sqrt(1 - (x - 1)**2)

# Convert to lambda for numerical integration
f = sp.lambdify((x, y), z, 'numpy')

# Compute the volume using double integration
# Ensure the integrand is correctly passed as a numerical function
def integrand(x, y):
    return f(x, y)

# Define the limits of integration
def y_lower_func(x):
    return -np.sqrt(1 - (x - 1)**2)

def y_upper_func(x):
    return np.sqrt(1 - (x - 1)**2)

# Integrate over the x-range and y-range
volume, _ = dblquad(integrand, 0, 2, y_lower_func, y_upper_func)
print("Volume of the solid:", volume)

# Visualization
X = np.linspace(0, 2, 100)
Y = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(X, Y)
Z = 4 - X*2 - Y*2

fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis')
plt.title('Paraboloid z = 4 - x^2 - y^2 inside the cylinder')
fig.colorbar(surf)

# Cylinder visualization
theta = np.linspace(0, 2 * np.pi, 100)
x_cyl = 1 + np.cos(theta)
y_cyl = np.sin(theta)
z_cyl = np.linspace(0, 4, 100)
X_cyl, Z_cyl = np.meshgrid(x_cyl, z_cyl)
Y_cyl, _ = np.meshgrid(y_cyl, z_cyl)

ax.plot_surface(X_cyl, Y_cyl, Z_cyl, color='orange', alpha=0.6)

plt.show()