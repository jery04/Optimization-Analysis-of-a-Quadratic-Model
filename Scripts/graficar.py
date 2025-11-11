from __future__ import annotations  # Mejora las anotaciones de tipo.
from typing import Callable, Optional  # Tipos para funciones y valores opcionales.
from max_descenso import maximo_descenso_optimo  # Algoritmo de descenso óptimo.
from newton import newton_quadratic  # Método de Newton para funciones cuadráticas.
import numpy as np  # Operaciones numéricas con arrays.
import matplotlib.pyplot as plt  # Gráficos y visualizaciones.
import matplotlib.colors as mcolors  # Manejo de colores en gráficos.

# -------------------------------------
# Métodos de graficación y traycetorias
# -------------------------------------
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

# -----------------------------------
# Prueba de graficación
# -----------------------------------
if __name__ == '__main__':
    # f(x, y) = (x + y - 7)^2 + (2x + y - 5)^2
    f = lambda X, Y: (X + Y - 7.0)**2 + (2.0 * X + Y - 5.0)**2

    # Intervalos de graficación
    xlim = (-500.0, 500.0)
    ylim = (-500.0, 500.0)

    # Graficar superficie 3D
    graficar_superficie(
        f,
        xlim=xlim,
        ylim=ylim,
        n=300,
        xlabel='x', ylabel='y', zlabel='f(x,y)',
        title='Superficie: f(x,y) = (x + y - 7)^2 + (2x + y - 5)^2',
        cmap='Spectral_r',
        elev=38, azim=-110,
        dist=20,   # vista alejada
        expand=1.0
    )

    # Parámetros del problema cuadrático
    Q = np.array([[10., 6.],
                  [6., 4.]])
    c = np.array([-34., -24.])
    r = 74.0  # Se usa sólo para evaluación de f(x_min), no para la trayectoria de contornos
    x0 = np.array([0., 0.])

    # Obtener resultados del algoritmo de máximo descenso
    x_min, info = maximo_descenso_optimo(Q, c, x0, tol=1e-9, max_iter=1000, verbose=False)

    # Obtener trayectoria desde el algoritmo de máximo descenso
    puntos_demo_raw = info.get("ws", info.get("xs"))
    puntos_demo = [tuple(row[:2]) for row in np.asarray(puntos_demo_raw, dtype=float)]
    
    # Graficar trayectoria sobre contornos
    graficar_trayectoria(
        puntos=puntos_demo,
        f=f,
        xlim=(x0[0] - 10.0, x0[0] + 10.0),
        ylim=(x0[1] - 10.0, x0[1] + 10.0),
        n=200,
        levels=35,
        xlabel='x', ylabel='y',
        title='Trayectoria sobre contornos de f(x,y)',
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

    # Método de Newton (cuadrático)
    x_newton, xs_newton = newton_quadratic(Q, c, x0)

    # Convertir trayectoria a lista de tuplas (x, y)
    puntos_newton = [tuple(row[:2]) for row in np.asarray(xs_newton, dtype=float)]

    # Graficar trayectoria de Newton sobre los mismos contornos
    graficar_trayectoria(
        puntos=puntos_newton,
        f=f,
        xlim=(x0[0] - 10.0, x0[0] + 10.0),
        ylim=(x0[1] - 10.0, x0[1] + 10.0),
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