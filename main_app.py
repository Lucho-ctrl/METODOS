"""Numerical Core Module for Fixed-Point Method.

This module provides the numerical algorithms and utilities for solving
systems of non-linear equations using the fixed-point iteration method.
It contains the core mathematical functions, expression parser, and
predefined examples used by the graphical interface.

Main Components:
    - punto_fijo_sistema: Fixed-point iteration algorithm with relaxation
    - norma_jacobiana: Jacobian computation using finite differences
    - build_G: Safe expression parser for user input
    - EJEMPLOS: Predefined example systems

Dependencies:
    - numpy: Numerical computations
    - math: Mathematical functions
"""

import numpy as np
import math


def punto_fijo_sistema(G, x0, tol=1e-8, max_iter=500, omega=1.0):
    """Fixed point iteration method for systems of equations.
    
    Implements the fixed-point iteration algorithm for solving systems
    of non-linear equations of the form x = G(x). Supports relaxation
    parameter omega to improve convergence.
    
    Args:
        G: Function G(x) that returns a numpy array [g1(x,y), g2(x,y)]
        x0: Initial approximation as numpy array
        tol: Convergence tolerance (default: 1e-8)
        max_iter: Maximum number of iterations (default: 500)
        omega: Relaxation parameter, 0 < omega <= 1 (default: 1.0)
    
    Returns:
        tuple: (solution, errors, history, iterations, converged)
            - solution: Final approximation as numpy array
            - errors: List of error values at each iteration
            - history: List of approximations at each iteration
            - iterations: Number of iterations performed
            - converged: Boolean indicating if convergence was achieved
    """
    x = x0.copy().astype(float)
    historial, errores = [x.copy()], []
    for k in range(1, max_iter + 1):
        Gx = G(x)
        x_nuevo = (1 - omega) * x + omega * Gx
        error = np.linalg.norm(x_nuevo - x, ord=np.inf)
        errores.append(error)
        x = x_nuevo
        historial.append(x.copy())
        if error < tol:
            return x, errores, historial, k, True
    return x, errores, historial, max_iter, False


def norma_jacobiana(G, x, h=1e-5):
    """Compute Jacobian norm of G at point x using finite differences.
    
    Calculates the Jacobian matrix of function G at point x using
    central finite differences, then returns its infinity norm.
    
    Args:
        G: Function G(x) that returns a numpy array
        x: Point at which to compute Jacobian as numpy array
        h: Step size for finite differences (default: 1e-5)
    
    Returns:
        tuple: (norm, jacobian_matrix)
            - norm: Infinity norm of the Jacobian matrix
            - jacobian_matrix: The computed Jacobian matrix
    """
    n = len(x)
    J = np.zeros((n, n))
    for j in range(n):
        xp, xm = x.copy(), x.copy()
        xp[j] += h
        xm[j] -= h
        J[:, j] = (G(xp) - G(xm)) / (2 * h)
    return np.linalg.norm(J, ord=np.inf), J


# Safe namespace for expression evaluation
SAFE_NS = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
SAFE_NS.update({
    "np": np,
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "abs": abs,
    "pi": math.pi,
    "e": math.e
})


def build_G(expr_g1, expr_g2):
    """Build G(x) function from string expressions.
    
    Parses string expressions for g1(x,y) and g2(x,y) and returns
    a callable function that evaluates them safely.
    
    Args:
        expr_g1: String expression for g1(x,y)
        expr_g2: String expression for g2(x,y)
    
    Returns:
        function: G(x_arr) that returns numpy array [g1(x,y), g2(x,y)]
    
    Raises:
        Exception: If expression evaluation fails
    """
    def G(x_arr):
        x, y = float(x_arr[0]), float(x_arr[1])
        ns = dict(SAFE_NS)
        ns.update({"x": x, "y": y})
        r1 = eval(expr_g1, {"__builtins__": {}}, ns)
        r2 = eval(expr_g2, {"__builtins__": {}}, ns)
        return np.array([float(r1), float(r2)])
    return G


# Predefined example systems
EJEMPLOS = {
    "Ejemplo 1: Cuadrático": {
        "desc": "x² + y = 1,  x + y² = 1",
        "g1": "sqrt(1 - y)",
        "g2": "sqrt(1 - x)",
        "x0": 0.5,
        "y0": 0.5,
        "omega": 1.0,
    },
    "Ejemplo 2: Círculo/Parábola": {
        "desc": "x²+y²=4  ∩  y=x²-1",
        "g1": "sqrt(y + 1)",
        "g2": "sqrt(4 - x**2)",
        "x0": 1.5,
        "y0": 1.0,
        "omega": 0.5,
    },
    "Ejemplo 3: Exponencial": {
        "desc": "x·eʸ=2,  y·eˣ=3",
        "g1": "2 / exp(y)",
        "g2": "3 / exp(x)",
        "x0": 0.5,
        "y0": 0.9,
        "omega": 0.5,
    },
}


def main():
    """Main entry point for the application.
    
    Imports the graphical interface from Interfaz_grafica module
    and launches the application.
    """
    from Interfaz_grafica import launch_interface
    app = launch_interface()
    app.mainloop()


if __name__ == "__main__":
    main()
