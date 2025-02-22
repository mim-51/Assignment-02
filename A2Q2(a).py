import numpy as np
import matplotlib.pyplot as plt

# Define the parametric equations for r(t)
def r(t):
    return np.array([5*np.cos(t), 4*np.sin(t)])

# Define the derivative r'(t)
def r_prime(t):
    return np.array([-5*np.sin(t), 4*np.cos(t)])

# Define the unit tangent vector T(t)
def unit_tangent(t):
    rp = r_prime(t)
    norm = np.linalg.norm(rp)
    return rp / norm

# Generate points for the curve C
t_values = np.linspace(0, 2*np.pi, 300)
curve_points = np.array([r(t) for t in t_values])

# Compute position and tangent vectors at t = pi/4 and t = pi
t1, t2 = np.pi/4, np.pi
r_t1, r_t2 = r(t1), r(t2)
rp_t1, rp_t2 = r_prime(t1), r_prime(t2)
T_t1, T_t2 = unit_tangent(t1), unit_tangent(t2)

# Plot the curve C
plt.figure()
plt.plot(curve_points[:, 0], curve_points[:, 1], label='Curve C', color='blue')

# Plot position vectors
plt.quiver(0, 0, r_t1[0], r_t1[1], angles='xy', scale_units='xy', scale=1, color='green', label='r(pi/4)')
plt.quiver(0, 0, r_t2[0], r_t2[1], angles='xy', scale_units='xy', scale=1, color='red', label='r(pi)')

# Plot unit tangent vectors
plt.quiver(r_t1[0], r_t1[1], T_t1[0], T_t1[1], angles='xy', scale_units='xy', scale=1, color='orange', label="T(pi/4)")
plt.quiver(r_t2[0], r_t2[1], T_t2[0], T_t2[1], angles='xy', scale_units='xy', scale=1, color='purple', label="T(pi)")

# Labels and grid

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Curve C and Unit Tangent Vectors')
plt.grid()
plt.axis('equal')
plt.show()

# Print computed values
print("Position Vectors:")
print("r(pi/4) =", r_t1)
print("r(pi) =", r_t2)
print("\nUnit Tangent Vectors:")
print("T(pi/4) =", T_t1)
print("T(pi) =", T_t2)