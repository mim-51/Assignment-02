import sympy as sp
x, y = sp.symbols('x y')
f = sp.sqrt(1 - x**2)
fx = sp.diff(f, x)
fy = sp.diff(f, y)
integrand = sp.sqrt(1 + fx*2 + fy*2)
area = sp.integrate(integrand, (x, 0, 1), (y, 0, 4))
print(area)