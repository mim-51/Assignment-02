from sympy import symbols, Eq, solve, diff

# Define variables
x, y, z, lambd = symbols('x y z lambda')

# Define the temperature function T(x, y, z)
T = 8*x**2 + 4*y*z - 16*z + 600

# Define the constraint g(x, y, z) = 0 (ellipsoid equation)
g = 4*x*2 + y*2 + 4*z*2 - 16

# Compute gradients
grad_T = [diff(T, var) for var in (x, y, z)]
grad_g = [diff(g, var) for var in (x, y, z)]

# Set up the system of equations for Lagrange multipliers
equations = [Eq(grad_T[i], lambd * grad_g[i]) for i in range(3)]
equations.append(Eq(g, 0))  # Add the constraint equation

# Solve the system
solutions = solve(equations, (x, y, z, lambd), dict=True)

# Print all solutions
print("All solutions:")
#for sol in solutions:
    #print("(x, y, z, lambda) =", (sol[x], sol[y], sol[z], sol[lambd]))

# Evaluate T(x, y, z) at the solutions
temperatures = [(sol[x], sol[y], sol[z], T.subs(sol)) for sol in solutions]

# Find the maximum temperature result
hottest_point = max(temperatures, key=lambda t: t[3])

# Print results
print("Hottest point (x, y, z):", (hottest_point[0], hottest_point[1], hottest_point[2]))
print("Maximum temperature:", hottest_point[3])