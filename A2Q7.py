#A2Q7
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

#(a)
# Define the variables
x, y = sp.symbols('x y')

# Define the temperature function
T = 10 - 8*x*2 - 2*y*2

# Compute the double integral
double_integral = sp.integrate(sp.integrate(T, (y,0,2)), (x,0,1))

# Compute the area of the region
area = (1 - 0) * (2 - 0)

# Compute the average temperature
average_temperature = double_integral / area

# Simplify the result
print("(a)The average temperature is:", sp.simplify(average_temperature),"°C")

#(b)
t = sp.symbols('t')

# Define the integrand
integrand = sp.cos(t) * sp.sin(t) + t**3

# Compute the differential arc length ds
dx_dt,dy_dt,dz_dt = -sp.sin(t),sp.cos(t),1
ds = sp.sqrt(dx_dt*2 + dy_dt*2 + dz_dt*2)

# Multiply the integrand by ds
total_integrand = integrand * ds

# Perform the definite integral from t = 0 to t = pi
line_integral = sp.integrate(total_integrand, (t, 0, sp.pi))

# Simplify the result
print("(b)The value of the line integral is:", sp.simplify(line_integral))

#(c)
r, theta, z, h = sp.symbols('r theta z h')

# Define the density function in cylindrical coordinates
rho = r**2

# Define the volume element in cylindrical coordinates
dV = r

# Multiply the density by the volume element
integrand = rho * dV

# Perform the triple integral over r, theta, and z
mass = sp.integrate(sp.integrate(sp.integrate(integrand, (r, 0, r)), (theta, 0, 2*sp.pi)), (z, -h/2, h/2))

# Simplify the result
print("(c)The mass of the cylinder is:", sp.simplify(mass))

#(d)

# Define the variables and the force field
x, y = sp.symbols('x y')
F_x = sp.exp(y)
F_y = x * sp.exp(y)

# i. Verify that the force field is conservative
# A force field is conservative if the curl is zero
curl_F = sp.diff(F_y, x) - sp.diff(F_x, y)
is_conservative = curl_F == 0
print(f"Is the force field conservative? {is_conservative}")

# ii. Find a potential function phi
# Integrate F_x with respect to x to find phi
phi_x = sp.integrate(F_x, x)

# Since phi_x and phi_y should be equal up to a constant, we can combine them
phi = phi_x + sp.integrate(F_y - sp.diff(phi_x, y), y)
print("(d)Potential function phi:",phi)

# iii. Find the work done by the field along the semicircular path C
# Since the field is conservative, the work done is the difference in the potential function
# at the endpoints

phi_1_0 = phi.subs({x: 1, y: 0})
phi_minus1_0 = phi.subs({x: -1, y: 0})
work_done = phi_minus1_0 - phi_1_0
print(f"Work done by the field: {work_done}")

# Plot the semicircular path C
theta = np.linspace(0, np.pi, 100)
x_path = np.cos(theta)
y_path = np.sin(theta)

plt.figure(figsize=(6, 6))
plt.plot(x_path, y_path, label='Semicircular Path C')
plt.scatter([1, -1], [0, 0], color='red', label='Endpoints (1,0) and (-1,0)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Semicircular Path from (1,0) to (-1,0)')
plt.legend()
plt.grid()
plt.axis('equal')
plt.show()