import sympy as sp
import numpy as np

# Part (a): Green's Theorem to find work done
x, y ,z= sp.symbols('x y z')
P = sp.exp(x) - y**3
Q = sp.cos(y) + x**3

# curl F = dQ/dx - dP/dy
curl_F = sp.diff(Q, x) - sp.diff(P, y)

# Unit circle area integral (double integral of curl_F over the disk)
r, theta = sp.symbols('r theta')
# r goes from 0 to 1, theta from 0 to 2pi
dA = r
integral = sp.integrate(sp.integrate(curl_F.subs({x: r*sp.cos(theta), y: r*sp.sin(theta)}) * dA, (r, 0, 1)), (theta, 0, 2*sp.pi))
print("(a) Work done by the force field:", integral)  # Answer: 3pi/2

# Part (b): Surface integral over the sphere x^2 + y^2 + z^2 = 1 using spherical coordinates
rho, phi, theta = sp.symbols('rho phi theta')
# x = rho*sin(phi)*cos(theta), y = rho*sin(phi)*sin(theta), z = rho*cos(phi)
x_sph = rho * sp.sin(phi) * sp.cos(theta)

# Jacobian for spherical coordinates: rho^2 * sin(phi)
jacobian = rho**2 * sp.sin(phi)

# x^2 in spherical coordinates
x_sphh= (rho * sp.sin(phi) * sp.cos(theta))**2

# Surface integral over sphere of radius 1 (rho = 1)
surface_integral = sp.integrate(sp.integrate((x_sphh * jacobian).subs(rho, 1), (phi, 0, sp.pi)), (theta, 0, 2*sp.pi))
print("(b) Surface integral over the sphere using spherical coordinates:", surface_integral)  # Answer: 4pi/3

# Part (c): Divergence Theorem for flux
#x, y, z = sp.symbols('x y z')
F = [x**3, y**3, z**2]

# div F = d(F1)/dx + d(F2)/dy + d(F3)/dz
div_F = sp.diff(F[0], x) + sp.diff(F[1], y) + sp.diff(F[2], z)

# Volume of the cylinder: r^2 <= 9, 0 <= z <= 2
R, Z = sp.symbols('R Z')
dv=R
volume_integral = sp.integrate(sp.integrate(sp.integrate(div_F.subs({x: R*sp.cos(theta), y: R*sp.sin(theta), z: Z}) * dv, (R, 0, 3)), (theta, 0, 2*sp.pi)), (Z, 0, 2))
print("(c) Outward flux using Divergence Theorem:", volume_integral)  # Answer: 279pi

# Part (d): Stokes' Theorem verification
# x, y, z = sp.symbols('x y z')
F = [2*z, 3*x, 5*y]

# Curl of F
curl_F = sp.Matrix([sp.diff(F[2], y) - sp.diff(F[1], z), sp.diff(F[0], z) - sp.diff(F[2], x), sp.diff(F[1], x) - sp.diff(F[0], y)])

# Dot with dS = k dA for the paraboloid z = 4 - x^2 - y^2
dA = 1
surface_integral = curl_F[2] * dA * sp.pi * 4  # Area of disk at z = 4 with radius 2
print("(d) Stokes' Theorem verification (surface integral):", surface_integral)  # Answer: 12pi

# Line integral verification using Stokes' Theorem
# Parametrize the circle x^2 + y^2 = 4
t = sp.symbols('t')
x_t = 2 * sp.cos(t)
y_t = 2 * sp.sin(t)
z_t = 0

# dr = (-2*sin(t), 2*cos(t)) dt
dx = sp.diff(x_t, t)
dy = sp.diff(y_t, t)
dz = sp.diff(z_t, t)

# F dot dr
F_t = [2*z_t, 3*x_t, 5*y_t]
dr = [dx, dy, dz]
#dot_product = sum(F_t[i] * dr[i] for i in range(3))
dot_product=np.dot(F_t,dr)

# Integrate from 0 to 2pi
line_integral = sp.integrate(dot_product, (t, 0, 2*sp.pi))
print("(d) Stokes' Theorem verification (line integral):", line_integral)  # Answer: 12pi
if line_integral==surface_integral:
     print ('verified')
else:
     print('Not verified')