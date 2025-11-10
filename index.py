from __future__ import annotations  # Mejora las anotaciones de tipo.
import numpy as np  # Operaciones numéricas con arrays.
from typing import Callable, Optional  # Tipos para funciones y valores opcionales.
import matplotlib.pyplot as plt  # Gráficos y visualizaciones.
import matplotlib.colors as mcolors  # Manejo de colores en gráficos.
from max_descenso import maximo_descenso_optimo  # Algoritmo de descenso óptimo.
from newton import newton_quadratic  # Método de Newton para funciones cuadráticas.

#-----------------------------------
# Métodos de optimización
#-----------------------------------

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

#-----------------------------------
# Graficar superficie y trayectorias
#-----------------------------------
def graficar_superficie(
        f: Callable[[np.ndarray, np.ndarray], np.ndarray],
        xlim: tuple[float, float] = (-3.0, 3.0),
        ylim: tuple[float, float] = (-3.0, 3.0),
        n: int = 100,
        xlabel: str = 'x', ylabel: str = 'y', zlabel: str = 'z', title: str = '',
        cmap: str = 'viridis', elev: Optional[float] = None, azim: Optional[float] = None,
        dist: Optional[float] = None, expand: float = 1.0
    ) -> None:
    """
    Grafica la superficie z = f(x, y) sin mostrar plano de corte visible.
    Muestra también el contorno proyectado sobre el plano XY (como en el ejemplo sin(R)/R).
    """
    # Ajuste del rango
    expand = float(expand)
    if expand < 1.0:
        expand = 1.0
    if expand != 1.0:
        cx = 0.5 * (xlim[0] + xlim[1])
        hx = 0.5 * (xlim[1] - xlim[0]) * expand
        xlim = (cx - hx, cx + hx)
        cy = 0.5 * (ylim[0] + ylim[1])
        hy = 0.5 * (ylim[1] - ylim[0]) * expand
        ylim = (cy - hy, cy + hy)

    # Malla
    x = np.linspace(*xlim, int(n))
    y = np.linspace(*ylim, int(n))
    X, Y = np.meshgrid(x, y)
    Z = np.asarray(f(X, Y), dtype=float)

    # Recorte invisible (los valores que superan 15000 se hacen NaN)
    Z[Z > 15000] = np.nan

    # Figura
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Superficie principal
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, linewidth=0, antialiased=True, alpha=0.95)

    # Contorno proyectado sobre el plano XY (sin mostrar plano)
    ax.contour(X, Y, Z, zdir='z', offset=np.nanmin(Z) - 500, cmap=cmap, linewidths=1.5)

    # Etiquetas
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(np.nanmin(Z) - 500, 15000)  # no muestra corte visible

    if title:
        ax.set_title(title)

    # Ajuste de cámara
    if elev is not None or azim is not None:
        ax.view_init(elev=elev or 30, azim=azim or -60)
    if dist is not None and hasattr(ax, 'dist'):
        try:
            ax.dist = float(dist)
        except Exception:
            pass

    fig.colorbar(surf, shrink=0.65, aspect=12, pad=0.1)
    plt.tight_layout()
    plt.show()

def graficar_trayectoria(
        puntos: np.ndarray | list[tuple[float, float]],
        f: Callable[[np.ndarray, np.ndarray], np.ndarray],
        xlim: tuple[float, float] = (-5.0, 10.0),
        ylim: tuple[float, float] = (-5.0, 10.0),
        n: int = 100,
        levels: int | list[float] = 30,
        xlabel: str = 'x', ylabel: str = 'y', title: str = 'Trayectoria sobre contornos',
        cmap: str = 'viridis',
    color_trayectoria: str = 'tab:blue',
    mostrar_linea: bool = True,
    color_pasos: Optional[str] = None,
    marcador_pasos: str = '.',
    tam_puntos: int = 44,
    alpha_puntos: float = 0.95,
        marcador_inicio: str = 'o',
        marcador_fin: str = 'X',
        tam_inicio: int = 90,
        tam_fin: int = 120,
        mostrar_etiquetas_nivel: bool = False
    ) -> None:
    """
    Grafica los contornos de z = f(x, y) en el plano XY y superpone la trayectoria
    definida por un conjunto de puntos conectados por una línea.

    Parámetros
    ----------
    puntos: array-like de forma (m, 2)
        Secuencia de puntos (x, y) que definen la trayectoria a dibujar.
    f: Callable[[ndarray, ndarray], ndarray]
        Función escalar de dos variables para generar el mapa de contornos.
    xlim, ylim: (min, max)
        Límites de los ejes X e Y para construir la malla de contorno.
    n: int
        Resolución de la malla (n x n).
    levels: int o lista de float
        Niveles de contorno a dibujar (como en matplotlib.contour).
    xlabel, ylabel, title: str
        Etiquetas y título del gráfico.
    cmap: str
        Colormap para los contornos.
    color_trayectoria: str
        Color de la línea que conecta los puntos (si mostrar_linea=True).
    mostrar_linea: bool
        Si True dibuja una línea que conecta los puntos en orden.
    color_pasos: Optional[str]
        Color de los puntos individuales (pasos) de la trayectoria. Si es None,
        se usará un color más oscuro derivado de 'color_trayectoria'.
    marcador_pasos: str
        Marcador para cada punto de la trayectoria (pasos intermedios).
    tam_puntos: int
        Tamaño de los puntos de la trayectoria (por defecto más grande para mejor visibilidad).
    alpha_puntos: float
        Transparencia (alpha) de los puntos de la trayectoria.
    marcador_inicio, marcador_fin: str
        Marcadores para el punto inicial y el punto final, respectivamente.
    tam_inicio, tam_fin: int
        Tamaños de los marcadores de inicio y fin.
    mostrar_etiquetas_nivel: bool
        Si True, añade etiquetas numéricas a las curvas de nivel.
    """

    # Convertir y validar puntos
    P = np.asarray(puntos, dtype=float)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError("'puntos' debe ser de forma (m, 2)")
    if P.shape[0] == 0:
        raise ValueError("'puntos' no puede estar vacío")

    # Malla del plano XY
    x = np.linspace(*xlim, int(n))
    y = np.linspace(*ylim, int(n))
    X, Y = np.meshgrid(x, y)
    Z = np.asarray(f(X, Y), dtype=float)

    # Figura 2D con contornos
    fig, ax = plt.subplots(figsize=(9, 7))
    cs = ax.contour(X, Y, Z, levels=levels, cmap=cmap)
    if mostrar_etiquetas_nivel:
        ax.clabel(cs, inline=True, fontsize=9, fmt='%g')

    # Dibuja línea completa si se solicita
    if mostrar_linea:
        ax.plot(P[:, 0], P[:, 1], color=color_trayectoria, linewidth=2.0, alpha=0.65, label='Trayectoria')

    # Dibuja pequeños puntos para cada paso
    if color_pasos is None:
        # Derivar un color más oscuro a partir del color de la trayectoria
        try:
            base_rgb = np.array(mcolors.to_rgb(color_trayectoria))
            factor = 0.6  # <1.0 oscurece
            darker_rgb = tuple(np.clip(base_rgb * factor, 0.0, 1.0))
            color_pasos_eff = darker_rgb
        except Exception:
            color_pasos_eff = color_trayectoria
    else:
        color_pasos_eff = color_pasos

    ax.scatter(P[:, 0], P[:, 1], s=tam_puntos, c=[color_pasos_eff], marker=marcador_pasos, alpha=alpha_puntos, edgecolors='none', label='Pasos')

    # Punto inicial y final con símbolos únicos
    ax.scatter(P[0, 0], P[0, 1], s=tam_inicio, c='tab:green', marker=marcador_inicio, edgecolors='k', linewidths=0.6, label='Inicio')
    ax.scatter(P[-1, 0], P[-1, 1], s=tam_fin, c='tab:red', marker=marcador_fin, edgecolors='k', linewidths=0.6, label='Fin')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.25)
    ax.legend(loc='best')

    plt.tight_layout()
    plt.show()

#-----------------------------------
# Pruebas de los métodos y graficación
#-----------------------------------
if __name__ == "__main__":
    # f(x, y) = (x + y - 7)^2 + (2x + y - 5)^2
    # En forma cuadrática: f(x) = 1/2 x^T Q x + c^T x + r
    Q = np.array([[10.0, 6.0],
                  [ 6.0, 4.0]])
    c = np.array([-34.0, -24.0])
    r = 74.0
    x0 = np.array([0.0, 0.0])  # punto inicial (0,0)

    # Función escalar para graficación
    f = lambda X, Y: (X + Y - 7.0)**2 + (2.0 * X + Y - 5.0)**2

    # Ejecutar Máximo Descenso (paso óptimo)
    x_min, info = maximo_descenso_optimo(Q, c, x0, tol=1e-9, max_iter=1000, verbose=True)
    f_xmin = 0.5 * x_min @ (Q @ x_min) + c @ x_min + r
    print("\n[Máximo descenso] Punto óptimo aproximado:", x_min)
    print("[Máximo descenso] Valor mínimo f(x*):", f_xmin)
    print("[Máximo descenso] Trayectoria (xs):\n", info.get("xs"))

    # ======================
    # Graficar superficie 3D de la función
    # ======================
    graficar_superficie(
        f,
        xlim=(-500.0, 500.0),
        ylim=(-500.0, 500.0),
        n=300,
        xlabel='x', ylabel='y', zlabel='f(x,y)',
        title='Superficie: f(x,y) = (x + y - 7)^2 + (2x + y - 5)^2',
        cmap='Spectral_r',
        elev=35, azim=-60,
        dist=20,   # vista alejada
        expand=1.0
    )

    # ======================
    # Trayectoria sobre contornos (Máximo descenso)
    # ======================
    puntos_descenso_raw = info.get("ws", info.get("xs"))
    puntos_descenso = [tuple(row[:2]) for row in np.asarray(puntos_descenso_raw, dtype=float)]
    graficar_trayectoria(
        puntos=puntos_descenso,
        f=f,
        xlim=(-5.0, 10.0),
        ylim=(-5.0, 10.0),
        n=200,
        levels=35,
        xlabel='x', ylabel='y',
        title='Trayectoria (Máximo descenso) sobre contornos de f(x,y)',
        cmap='viridis',
        color_trayectoria='orange',
        mostrar_linea=True,
        marcador_pasos='.',
        tam_puntos=48,
        alpha_puntos=0.95,
        marcador_inicio='o',
        marcador_fin='s',
        mostrar_etiquetas_nivel=True
    )

    # ======================
    # Método de Newton (cuadrático)
    # ======================
    x_newton, xs_newton = newton_quadratic(Q, c, x0)
    f_xnewton = 0.5 * x_newton @ (Q @ x_newton) + c @ x_newton + r
    print("\n[Newton] Punto óptimo (una iteración):", x_newton)
    print("[Newton] Valor mínimo f(x*):", f_xnewton)

    # Graficar trayectoria de Newton sobre los mismos contornos
    puntos_newton = [tuple(row[:2]) for row in np.asarray(xs_newton, dtype=float)]
    graficar_trayectoria(
        puntos=puntos_newton,
        f=f,
        xlim=(-5.0, 10.0),
        ylim=(-5.0, 10.0),
        n=200,
        levels=35,
        xlabel='x', ylabel='y',
        title='Trayectoria (Newton) sobre contornos de f(x,y)',
        cmap='viridis',
        color_trayectoria='tab:purple',
        mostrar_linea=True,
        marcador_pasos='o',
        tam_puntos=70,
        alpha_puntos=0.95,
        marcador_inicio='o',
        marcador_fin='s',
        mostrar_etiquetas_nivel=True
    )
    