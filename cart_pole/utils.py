import matplotlib.pyplot as plt
import numpy as np

def phase_portrait(
    f,
    x_range=[1, 1],
    cmap='gray',
    contour=False,
    cs_func=False,
    size=(7, 5),
    density=0.95,
    draw_grid=False,
):

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
    plt.title('Phase Portrait')

    if contour:
        plt.contourf(x1_span, x2_span, dist, cmap=cmap, alpha=0.15)

    plt.streamplot(x1_span, x2_span, dx1, dx2, arrowsize=1.2,   density=density, color=dist,
               cmap=cmap, linewidth=lw, arrowstyle='->')  # ,color=L, cmap='autumn', linewidth = lw)

    plt.xlabel(r'State  $x_1$')
    plt.ylabel(r'State  $x_2$')

    plt.xlim([-x1_max, x1_max])
    plt.ylim([-x2_max, x2_max])
    if draw_grid:
        plt.grid(color='black', linestyle='--', linewidth=1.0, alpha=0.3)
        plt.grid(True)
    plt.tight_layout()
    # show()

    return None


def elipse(x, P, r):
    return P[0,0]*x[0]**2 + 2*P[1,0]*x[1]*x[0] + P[1,1]*x[1]**2 - r


def draw_elipse(
    x_bounds=[-4, 4], 
    f=elipse,
    cs_func=False,
    args=tuple(),
    plt_args=dict()
):
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
    
    plt.contour(X1, X2, ROA, [0], **plt_args)


def sampling_roa(f, V, nablaV, x_bounds, N=100):
    x_min, x_max = x_bounds
    x_range = np.array(x_max) - np.array(x_min)
    n = np.shape(x_min)[0]

    c = np.inf
    ct = []

    for i in range(N):
        x_i = x_min + x_range*np.random.rand(n)
        dV_i = np.array(nablaV(x_i))@ np.array(f(x_i))
        V_i = V(x_i)
        if dV_i >= 0 and V_i <= c:
            c = V_i
            ct.append(c)  
    
    return c, ct