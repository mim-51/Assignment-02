import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import dblquad

# Define variables
x, y = sp.symbols('x y')

# Surface z = sqrt(4 - x^2)
z = sp.sqrt(4 - x**2)

# Partial derivative dz/dx
z_x = sp.diff(z, x)

# Surface area integrand (z does not depend on y, so dz/dy = 0)
integrand = sp.sqrt(1 + z_x**2)

# Convert to lambda function for numerical integration
f = sp.lambdify(x, integrand, 'numpy')

# Compute the surface area
surface_area = dblquad(lambda y, x: f(x), 0, 1, lambda x: 0, lambda x: 4)[0]
print("Surface area:", surface_area)

# Visualization
X = np.linspace(0, 1, 100)
Y = np.linspace(0, 4, 100)
X, Y = np.meshgrid(X, Y)
Z = np.sqrt(4 - X**2)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis')
plt.title('Surface plot of z = sqrt(4 - x^2)')
fig.colorbar(surf)
plt.show()