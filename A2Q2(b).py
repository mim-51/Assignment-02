import numpy as np
import matplotlib.pyplot as plt

# Arc length parametrization
def helix(s):
    x = np.cos(s / np.sqrt(2))
    y = np.sin(s / np.sqrt(2))
    z = s / np.sqrt(2)
    return x, y, z

# Generate points for the helix
s_values = np.linspace(0, 20, 500)
x_values = np.cos(s_values / np.sqrt(2))
y_values = np.sin(s_values / np.sqrt(2))
z_values = s_values / np.sqrt(2)

# Point for s = 10
s_10 = 10
x_10, y_10, z_10 = helix(s_10)

print(f"point at s=10: ({x_10:.3f},{y_10:.3f}, {z_10:.3f})")

# Plot the helix
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_values, y_values, z_values, label="Helix", color='b')

# Mark the point at s = 10
ax.scatter(x_10, y_10, z_10, color='r', label='the bug')

# Labels and title
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.set_title("Circular Helix with Point at s=10")

# Show legend
ax.legend()

# Show plot
plt.show()