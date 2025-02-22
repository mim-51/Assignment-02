import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
t, x, y,z, theta = sp.symbols('t x y z theta')
r1 = sp.Matrix([sp.exp(t), sp.exp(t) * sp.cos(t), sp.exp(t) * sp.sin(t)])
r2 = sp.Matrix([2 * sp.cos(t), 3 * sp.sin(t), 0])



def compute_frenet_serret(r):
    r_prime = r.diff(t)
    r_double_prime = r_prime.diff(t)
    r_triple_prime = r_double_prime.diff(t)
    T = r_prime / r_prime.norm()
    T_prime=T.diff(t)
    N=T_prime/T_prime.norm()
    #N = (r_double_prime - (r_double_prime.dot(T)) * T) / (r_double_prime - (r_double_prime.dot(T)) * T).norm()
    B = T.cross(N)
    
    kappa = (r_prime.cross(r_double_prime)).norm() / (r_prime.norm() ** 3)
    tau = (r_prime.cross(r_double_prime)).dot(r_triple_prime) / (r_prime.cross(r_double_prime)).norm() ** 2
    return T, N, B, kappa, tau



T1, N1, B1, kappa1, tau1 = compute_frenet_serret(r1)
T2, N2, B2, kappa2, tau2 = compute_frenet_serret(r2)



T1_at_0 = T1.subs(t, 0)
N1_at_0 = N1.subs(t, 0)
B1_at_0 = B1.subs(t, 0)
kappa1_at_0 = kappa1.subs(t, 0)
tau1_at_0 = tau1.subs(t, 0)



T2_at_2pi = T2.subs(t, 2 * np.pi)
N2_at_2pi = N2.subs(t, 2 * np.pi)
B2_at_2pi = B2.subs(t, 2 * np.pi)
kappa2_at_2pi = kappa2.subs(t, 2 * np.pi)
tau2_at_2pi = tau2.subs(t, 2 * np.pi)



print('-' * 100)
print('Question - 3 (a)(i) for t = 0')
print('-' * 100)
print(f'Tangent at t = 0: {T1_at_0}')
print(f'Normal at t = 0:  {N1_at_0}')
print(f'Binormal at t = 0: {B1_at_0}')
print(f'Curvature at t = 0: {kappa1_at_0}')
print(f'Torsion at t = 0: {tau1_at_0}')


print('-' * 100)
print('Question - 3 (a)(ii) for t = 2*pi')
print('-' * 100)
print(f'Tangent at t = 2π: {T2_at_2pi}')
print(f'Normal at t = 2π:  {N2_at_2pi}')
print(f'Binormal at t = 2π: {B2_at_2pi}')
print(f'Curvature at t = 2π: {kappa2_at_2pi}')
print(f'Torsion at t = 2π: {tau2_at_2pi}')



print("""
Comments for Curve 2 (r2) at t = 2π:
- The tangent vector (T2_at_2pi) shows the direction in which the curve is moving at t = 2π.
- The normal vector (N2_at_2pi) tells us how the curve is bending at this point.
- The binormal vector (B2_at_2pi) gives us the orientation of the curve in 3D space.
- The curvature (kappa2_at_2pi) tells us how sharply the curve bends at t = 2π.
- The torsion (tau2_at_2pi) is zero, meaning the curve does not twist at t = 2π and lies in a plane.
""")



t_values = np.linspace(0, 2 * np.pi, 100)
kappa1_values = [kappa1.subs(t, val) for val in t_values]
kappa2_values = [kappa2.subs(t, val) for val in t_values]

X = np.linspace(-2, 0, 200)
Y = np.linspace(0, 2, 200)
X, Y = np.meshgrid(X, Y)
D_T_values = np.zeros_like(X, dtype=float)



for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        grad_eval = np.array([sp.diff(3 * x**2 * y, x).subs({x: X[i, j], y: Y[i, j]}), sp.diff(3 * x**2 * y, y).subs({x: X[i, j], y: Y[i, j]})], dtype=float)
        D_T_values[i, j] = np.dot(grad_eval, np.array([-1, -1/2]) / np.linalg.norm([-1, -1/2]))




fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].plot(t_values, kappa1_values, label='Curve 1 (r1)', color='blue')
axes[0].plot(t_values, kappa2_values, label='Curve 2 (r2)', color='red')
axes[0].set_xlabel('t')
axes[0].set_ylabel('Curvature (κ)')
axes[0].set_title('Curvature of the Curves')
axes[0].legend()
axes[0].grid(True)




ax = fig.add_subplot(122, projection='3d')
surf = ax.plot_surface(X, Y, D_T_values, cmap='viridis', edgecolor='none')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Directional Derivative')
ax.set_title('Directional Derivative of T(x, y)')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
plt.tight_layout()
plt.show()



f = y**2 * sp.cos(x - y)
f_x = sp.diff(f, x)
f_y = sp.diff(f, y)
f_xx = sp.diff(f_x, x)
f_yy = sp.diff(f_y, y)
f_xy = sp.diff(f_x, y)
f_yx = sp.diff(f_y, x)
laplace_eq = f_xx + f_yy
cauchy_riemann_eq1 = f_x - f_y
cauchy_riemann_eq2 = f_x + f_y



laplace_verified = sp.simplify(laplace_eq) == 0
cauchy_riemann_verified = sp.simplify(cauchy_riemann_eq1) == 0 and sp.simplify(cauchy_riemann_eq2) == 0
identity_verified = sp.simplify(f_xy - f_yx) == 0



print(f'Laplace’s equation: {laplace_eq}')
if laplace_verified:
    print("Laplace's equation is verified.")
else:
    print("Laplace's equation is NOT verified.")
print(f'Cauchy-Riemann equations: {cauchy_riemann_eq1}, {cauchy_riemann_eq2}')
if cauchy_riemann_verified:
    print("Cauchy-Riemann equations are verified.")
else:
    print("Cauchy-Riemann equations are NOT verified.")
print(f'Identity f_xy = f_yx: {sp.simplify(f_xy - f_yx)}')
if identity_verified:
    print("The identity f_xy = f_yx is verified.")
else:
    print("The identity f_xy = f_yx is NOT verified.")
print('-' * 100)



w = sp.sqrt(x**2 + y**2 + z**2)
x_expr = sp.cos(theta)
y_expr = sp.sin(theta)
z_expr = sp.tan(theta)
w_expr = w.subs({x: x_expr, y: y_expr, z: z_expr})
dw_dtheta = sp.diff(w_expr, theta)
dw_dtheta_simplified = sp.simplify(dw_dtheta.subs({theta: sp.pi / 4}))



print(f'dw/dθ at θ = π/4: {dw_dtheta_simplified}')
print('-' * 100)
T = 3 * x**2 * y
Tx = sp.diff(T, x)
Ty = sp.diff(T, y)
grad_T = sp.Matrix([Tx, Ty])
point = {x: -1, y: -3/2}
grad_T_val = grad_T.subs(point)
dir_vector = np.array([-1, -1/2])
unit_vector = dir_vector / np.linalg.norm(dir_vector)
D_T = grad_T_val.dot(sp.Matrix(unit_vector))



print(f'Gradient of T at (-1, -3/2): {grad_T_val}')
print(f'Directional derivative in the direction (-1, -1/2): {D_T}')
print('-' * 100)

