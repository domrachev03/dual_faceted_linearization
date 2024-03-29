import matplotlib.pyplot as plt
import numpy as np


def phase_portrait(
    f,
    ax=None,
    x_range=[1, 1],
    cmap='gray',
    contour=False,
    cs_func=False,
    size=(7, 5),
    density=0.95,
    draw_grid=False,
):
    if ax is None:
        _, ax = plt.subplots()
    
    x1_max, x2_max = x_range
    x1_span = np.arange(-1.1*x1_max, 1.1*x1_max, 0.1)
    x2_span = np.arange(-1.1*x2_max, 1.1*x2_max, 0.1)
    x_grid = np.array(np.meshgrid(x1_span, x2_span))
    x1_grid, x2_grid = x_grid
    
    dx = np.array([
        [
            f(x_grid[:, i, k]) for k in range(len(x1_grid))
        ] for i in range(len(x2_grid))
    ])[:, :, :, 0]
    
    dx1, dx2 = dx[:, :, 0], dx[:, :, 1]

    dist = (x1_grid**2 + x2_grid**2)**0.5
    lw = 0.8*(2*dist + dist.max()) / dist.max()

    # figure(figsize=size)
    ax.set_title('Phase Portrait')

    if contour:
        ax.contourf(x1_span, x2_span, dist, cmap=cmap, alpha=0.15)

    ax.streamplot(x1_span, x2_span, dx1, dx2, arrowsize=1.2,   density=density, color=dist, cmap=cmap, linewidth=lw, arrowstyle='->')  # ,color=L, cmap='autumn', linewidth = lw)

    ax.set_xlabel(r'State  $x_1$')
    ax.set_ylabel(r'State  $x_2$')

    ax.set_xlim([-x1_max, x1_max])
    ax.set_ylim([-x2_max, x2_max])
    if draw_grid:
        ax.set_grid(color='black', linestyle='--', linewidth=1.0, alpha=0.3)
        ax.set_grid(True)
    
    return None


def elipse(x, P, r):
    return P[0,0]*x[0]**2 + 2*P[1,0]*x[1]*x[0] + P[1,1]*x[1]**2 - r


def draw_elipse(
    x_bounds=[-4, 4],
    ax=None,  
    f=elipse,
    cs_func=False,
    args=tuple(),
    plt_args=dict()
):
    if ax is None:
        fig, ax = plt.subplots()

    x_min, x_max = x_bounds
    delta = 0.025
    x1range = np.arange(x_min, x_max, delta)
    x2range = np.arange(x_min, x_max, delta)
    
    X1, X2 = np.meshgrid(x1range, x2range)
    X = np.array([X1, X2])

    if not cs_func:
        ROA = f(X, *args)
    else:
        ROA = np.array([
            [f(X[:, i, k], *args) for k in range(len(x2range))] for i in range(len(x1range))
        ])[:, :, 0, 0]
    
    ax.contour(X1, X2, ROA, [0], **plt_args)


def sampling_roa(f, V, dV, x_bounds, N=100):
    x_min, x_max = x_bounds
    x_range = np.array(x_max) - np.array(x_min)
    n = np.shape(x_min)[0]

    c = np.inf
    ct = []

    for i in range(N):
        x_i = x_min + x_range*np.random.rand(n)
        dV_i = dV(x_i)
        V_i = V(x_i)
        if V_i < 0:
            continue
        if dV_i >= 0 and V_i <= c:
            c = V_i
            ct.append(c)  
    
    return c, np.array(ct)