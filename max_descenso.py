import numpy as np

def maximo_descenso_optimo(Q, c, x0, tol=1e-8, max_iter=1000, verbose=False):
    """
    Máximo descenso para f(x) = 1/2 x^T Q x + c^T x + r
    usando el paso analítico óptimo alpha_k = (g^T g) / (g^T Q g).
    NOTA: La expresión alpha_k fue previamente hallada de forma analítica (en el LaTeX).

    Parámetros:
        Q : ndarray (n,n)  -- matriz simétrica
        c : ndarray (n,)   -- vector
        x0: ndarray (n,)   -- punto inicial
        tol: float         -- tolerancia para ||gradiente||
        max_iter: int      -- número máximo de iteraciones
        verbose: bool      -- imprimir o no el progreso

    Retorna:
        x : ndarray (n,)   -- punto mínimo (x1, ..., xn) en nuestro modelo x = (x1, x2)
        info : dict        -- historial (opcional)
    """
    # --- Validaciones simples ---
    Q = np.array(Q, dtype=float)
    c = np.array(c, dtype=float)
    x = np.array(x0, dtype=float)

    # --- Dimensiones ---
    n = x.shape[0]
    assert Q.shape == (n, n)
    assert c.shape == (n,)

    # --- Comprobar simetría numérica ---
    if not np.allclose(Q, Q.T, atol=1e-12):
        Q = 0.5 * (Q + Q.T)

    # Historial para análisis
    iters = []
    fvals = []
    alphas = []
    norms = []
    xs = []  # almacenará el vector x de cada iteración (incluye inicio y fin)

    # Guardar x inicial
    xs.append(x.copy())

    # --- Función objetivo ---
    def f_val(x_vec):
        return 0.5 * x_vec @ (Q @ x_vec) + c @ x_vec

    # --- Iteraciones ---
    for k in range(1, max_iter + 1):
        g = Q @ x + c                      # gradiente g_k = Q x_k + c
        gn = np.linalg.norm(g)             # ||g_k||

        iters.append(k)
        fvals.append(f_val(x))
        norms.append(gn)

        if verbose:
            print(f"Iter {k:3d}: ||g|| = {gn:.3e}, f = {fvals[-1]:.6e}")

        # Criterio de parada
        if gn < tol:
            if verbose:
                print("Criterio de parada alcanzado (||g|| < tol).")
            # Guardar x final antes de salir (si se cumple criterio de parada antes de actualizar)
            xs.append(x.copy())
            break

        Qg = Q @ g                         # producto Q g_k (reutilizable)
        denom = g @ Qg                     # g^T Q g

        # Seguridad numérica: evitar división por cero / denominador no positivo
        if denom <= 0:
            raise ValueError(
                "Denominador g^T Q g ≤ 0. Q puede no ser definida positiva o g≈0."
            )

        alpha = (g @ g) / denom            # alpha_k óptimo
        alphas.append(alpha)

        # Actualización
        x = x - alpha * g
        xs.append(x.copy())

    # Guardar x después de la actualización
    xs.append(x.copy())

    info = {
        "iters": np.array(iters),
        "fvals": np.array(fvals),
        "alphas": np.array(alphas),
        "norms": np.array(norms),
    }
    # Añadir trayectorias de x (cada fila corresponde a una iteración, incluyendo inicial y final)
    info["xs"] = np.vstack(xs)

    # Redondear cada elemento de x (evitar errores numéricos)
    x = np.round(x, decimals=2)

    return x, info

# -----------------------------------
# Finalmente, del papel al procesador
# -----------------------------------
if __name__ == "__main__":
    Q = np.array([[10., 6.],
                  [6., 4.]])
    c = np.array([-34., -24.])
    r = 74.0
    x0 = np.array([0., 0.])

    x_min, info = maximo_descenso_optimo(Q, c, x0, tol=1e-9, max_iter=1000, verbose=True)
    print("\nResultado numérico:", x_min)
    print("Valor mínimo f(x_min):", 0.5 * x_min @ (Q @ x_min) + c @ x_min + r)
    print(info["xs"])
