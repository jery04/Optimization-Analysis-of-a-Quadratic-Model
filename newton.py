import numpy as np

def newton_quadratic(Q, c, x0=None):
    """
    Método de Newton para función cuadrática
    f(x) = 1/2 x^T Q x + c^T x + r.

    Para este caso, una sola iteración produce el óptimo:
        x* = -Q^{-1} c

    Parámetros
    - Q: ndarray (n, n), matriz simétrica e invertible.
    - c: ndarray (n,), vector.
    - x0: ndarray (n,), punto inicial opcional (no afecta el resultado final en el caso cuadrático).

    Retorna
    - x1: ndarray (n,), punto óptimo (en una iteración, x1 = -Q^{-1} c).
    - info: ndarray (2, n), arreglo con los puntos [x0, x1] para formar una trayectoria mínima.
    """
    n = Q.shape[0]
    if x0 is None:
        x0 = np.zeros(n)
    b = - (Q @ x0 + c)        # -grad f(x0)
    # Resolver Q p = b  (p = paso de Newton)
    p = np.linalg.solve(Q, b)
    x1 = x0 + p               # en una iteración, x1 = -Q^{-1} c
    info = np.vstack([x0, x1])  # trayectoria: inicio y final
    return x1, info


# -----------------------------------
# Finalmente, del papel al procesador
# -----------------------------------
if __name__ == "__main__":
    Q = np.array([[10., 6.],[6., 4.]])  
    c = np.array([-34., -24.])
    x_star, xs = newton_quadratic(Q, c)
    print("x* =", x_star)
    print("Trayectoria xs =\n", xs)