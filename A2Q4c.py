import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Define variables and function
x, y = sp.symbols('x y')
f = 4*x*y - x*4 - y*4

# Compute first-order partial derivatives
f_x = sp.diff(f, x)
f_y = sp.diff(f, y)

# Find critical points
critical_points = sp.solve([f_x, f_y], (x, y), dict=True)

# Compute second-order partial derivatives
f_xx = sp.diff(f_x, x)
f_yy = sp.diff(f_y, y)
f_xy = sp.diff(f_x, y)

# Compute Hessian determinant D = f_xx * f_yy - (f_xy)^2
D = f_xx * f_yy - f_xy**2

# Filter only real critical points
numeric_critical_points = []
print("Critical Points and Classification:")

for point in critical_points:
    x_val, y_val = point[x], point[y]

    if x_val.is_real and y_val.is_real:
        x_val = float(x_val)
        y_val = float(y_val)

        D_val = float(D.subs({x: x_val, y: y_val}))
        f_xx_val = float(f_xx.subs({x: x_val, y: y_val}))

        if D_val > 0 and f_xx_val > 0:
            classification = 'Local Minimum'
        elif D_val > 0 and f_xx_val < 0:
            classification = 'Local Maximum'
        elif D_val < 0:
            classification = 'Saddle Point'
        else:
            classification = 'Test Inconclusive'

        print(f"Point: ({x_val}, {y_val}), D: {D_val}, f_xx: {f_xx_val}, Classification: {classification}")
        numeric_critical_points.append((x_val, y_val))

# Create a grid of points for 3D plotting
x_vals = np.linspace(-2, 2, 400)
y_vals = np.linspace(-2, 2, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z = 4*X*Y - X*4 - Y*4

# Plot the function surface
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)

# Mark critical points on the plot
for point in numeric_critical_points:
    x_p, y_p = point
    z_p = 4*x_p*y_p - x_p*4 - y_p*4
    ax.scatter(x_p, y_p, z_p, color='r', s=100 , label="Critical Point")

plt.legend()
plt.show(block=True)  # Ensures the plot stays open in VS Code