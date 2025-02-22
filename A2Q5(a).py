import numpy as np
import matplotlib.pyplot as plt

# Given ellipsoid equation: F(x, y, z) = x^2 + 4y^2 + z^2 - 18 = 0

# Step 1: Compute the gradient (normal vector to the surface)
def gradient_F(x, y, z):
    Fx = 2 * x
    Fy = 8 * y
    Fz = 2 * z
    return np.array([Fx, Fy, Fz])

# Point of tangency
x0, y0, z0 = 1, 2, 1

# Compute normal vector at (1,2,1)
normal_vector = gradient_F(x0, y0, z0)
Fx, Fy, Fz = normal_vector

# Step 2: Equation of the Tangent Plane: Fx(x - x0) + Fy(y - y0) + Fz(z - z0) = 0
tangent_plane_equation = f"{Fx}(x - {x0}) + {Fy}(y - {y0}) + {Fz}*(z - {z0}) = 0"

# Step 3: Parametric Equations of the Normal Line
t = np.linspace(-5, 5, 10)  # Parameter range for visualization
x_line = x0 + Fx * t
y_line = y0 + Fy * t
z_line = z0 + Fz * t

parametric_equations = (f"x = {x0} + {Fx}t", f"y = {y0} + {Fy}t", f"z = {z0} + {Fz}t")

# Step 4: Compute the angle with the xy-plane
normal_xy = np.array([0, 0, 1])  # Normal to xy-plane

# Compute dot product and magnitudes
dot_product = np.abs(np.dot(normal_vector, normal_xy))
magnitude_normal = np.linalg.norm(normal_vector)
magnitude_xy = np.linalg.norm(normal_xy)

# Compute cosine of the angle
cos_theta = dot_product / (magnitude_normal * magnitude_xy)
theta_rad = np.arccos(cos_theta)  # Angle in radians
theta_deg = np.degrees(theta_rad)  # Convert to degrees

# Display results
print("Equation of the Tangent Plane:")
print(tangent_plane_equation)

print("\nParametric Equations of the Normal Line:")
for eq in parametric_equations:
    print(eq)

print(f"\nAcute Angle with xy-plane: {theta_deg:.2f} degrees")

 #Visualization
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Generate grid for ellipsoid surface
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = np.sqrt(18) * np.outer(np.cos(u), np.sin(v))
y = np.sqrt(18) * np.outer(np.sin(u), np.sin(v)) / 2
z = np.sqrt(18) * np.outer(np.ones(np.size(u)), np.cos(v))

# Plot the ellipsoid surface
ax.plot_surface(x, y, z, color='yellow', alpha=0.6, rstride=4, cstride=4)

# Plot the tangent plane at (1,2,1)
xx, yy = np.meshgrid(np.linspace(-5, 5, 30), np.linspace(-5, 5, 30))
zz_tangent = (18 - xx*2 - 4*yy*2)  # Solving for z from the ellipsoid equation
ax.plot_surface(xx, yy, zz_tangent, alpha=0.6, color='blue')

# Plot the normal line
ax.plot(x_line, y_line, z_line, label='Normal Line', color='red', lw=2)

# Plot the point of tangency
ax.scatter(x0, y0, z0, color='green', s=100, label="Tangency Point (1,2,1)")

# Labels and plot settings
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('Ellipsoid, Tangent Plane, and Normal Line')
ax.legend()

plt.show()

# Visualization
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Generate grid for tangent plane
xx, yy = np.meshgrid(np.linspace(-5, 5, 30), np.linspace(-5, 5, 30))
zz = (18 - xx*2 - 4*yy*2)  # Solving for z from the ellipsoid equation

# Plot the ellipsoid surface (for context)
ax.plot_surface(xx, yy, zz, alpha=0.3, cmap='viridis')

# Plot the tangent plane at (1,2,1)
zz_tangent = (18 - xx*2 - 4*yy*2)  # Tangent plane z-values
ax.plot_surface(xx, yy, zz_tangent, alpha=0.6, color='blue')

# Plot the normal line
ax.plot(x_line, y_line, z_line, label='Normal Line', color='red', lw=2)

# Plot the point of tangency
ax.scatter(x0, y0, z0, color='green', s=100, label="Tangency Point (1,2,1)")

# Labels and plot settings
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('Ellipsoid, Tangent Plane, and Normal Line')
ax.legend()

plt.show()