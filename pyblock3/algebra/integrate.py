
"""
Integrate functions at module level.
"""

# apply exp(a) * b
def rk4_apply(a, b):
    k1 = a @ b
    k2 = a @ (0.5 * k1 + b)
    k3 = a @ (0.5 * k2 + b)
    k4 = a @ (k3 + b)
    return b + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6
