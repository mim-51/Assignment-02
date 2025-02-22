import sympy as sp

x, y, z = sp.symbols('x y z')

f = 4 - x*2 - y*2
g = (x - 1)*2 + y*2 - 1

z_min, z_max = 0, f
x_min, x_max = 0, 2

y_min = -sp.sqrt(1 - (x - 1)**2)
y_max = sp.sqrt(1 - (x - 1)**2)

int_1 = sp.integrate(1, (z, z_min, z_max)).simplify()

volume = sp.integrate(sp.integrate(int_1, (y, y_min, y_max)), (x, x_min, x_max))
print("Volume of the region:", volume)