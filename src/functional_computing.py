import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import  FuncAnimation
import warnings
warnings.filterwarnings('ignore')


def regular_attractor(v, t, a, idvars, d_st, w_st):

    a = a
    b, c, d = idvars

    # x
    w_0, d_0 = w_st[0], d_st[0]

    # y
    w_1, d_1 = w_st[1], d_st[1]
    w_2, d_2 = w_st[2], d_st[2]

    # z
    w_3, d_3 = w_st[3], d_st[3]

    # vector assignment:
    x, y, z = v[0], v[1], v[2]

    # ODEs:
    dx_dt = a * x - b * x ** 2 - (w_0 * x * y) / (x + d_0)
    dy_dt = -c * y + (w_1 * x * y) / (x + d_1) - (w_2 * y * z) / (y + d_2)
    dz_dt = d * z ** 2 - (w_3 * z ** 2) / (y + d_3)

    return [dx_dt, dy_dt, dz_dt]


def init_cond(a):

    x0 = a / 0.12 - 5 + np.sqrt((a + 0.6) ** 2 - 1.52) / 0.12
    y0 = 6.31579
    z0 = -40.3 + 80.6 * (x0 / (x0 + 10))

    return [x0, y0, z0]


def integrate(t, a, idvars, d_st, w_st, model=regular_attractor):
    v = odeint(model, init_cond(a) , t, args=(a, idvars, d_st, w_st)) # [17.494, 6.31579, 10.985]init_cond(a)[15, 6.31579, 8]
    X, Y, Z = v[:, 0], v[:, 1], v[:, 2]
    return X, Y, Z


#def updata_variability(i, t, a_vec, prey, predator, apex, time_text): #, time_text
#    X, Y, Z = integrate(t, a_vec[i], tuple((0.06, 1, 0.038)), [10, 10, 10, 20], [1, 2, 0.405, 1])
#
#    print(X)
#    prey.set_xdata(t)
#    predator.set_xdata(t)
#    apex.set_xdata(t)
#
#    prey.set_ydata(X)
#    predator.set_ydata(Y)
#    apex.set_ydata(Z)
#
#    time_text.set_text(f"$a={a_vec[i].round(3)}$")
#
#    return prey,predator,apex

#def plot_animated_dependencies():
#    fig, ax = plt.subplots()
#
#    T = np.linspace(0, 1000, 1000)
#    prey, = ax.plot([],[], color='green')
#    predator, = ax.plot([],[], color='blue')
#    apex, = ax.plot([],[], color='orange')
#    time_text = ax.text(0.1, 0.95, "", color='red')
#    a_v = np.linspace(1., 1.9, 1000)
#    animation = FuncAnimation(fig, updata_variability, frames=1000, fargs=(T, a_v, prey, predator, apex, time_text))
#    plt.show()
#    pass

def plot_animation_variablility():
    t = np.linspace(0, 1000, 10000)
    idstat = tuple((0.06, 1, 0.038))
    d_stat = [10, 10, 10, 20]
    w_stat = [1, 2, 0.405, 1]

    for a in np.linspace(.9, 1.9, 600):
        plt.clf()
        X, Y, Z = integrate(t, a, idstat, d_stat, w_stat, model=regular_attractor)
        plt.plot(t, X, color='green', label=f'prey, $a={a.round(4)}$')
        plt.plot(t, Y, color='orange', label=f'predator, $a={a.round(4)}$')
        plt.plot(t, Z, color='blue', label=f'top predator, $a={a.round(4)}$')
        plt.xlabel('$t$')
        plt.ylabel('Population density')
        plt.legend(bbox_to_anchor=(.95, 1.0), loc='upper left')
        plt.pause(0.001)
    plt.show()
    pass

def plot_3d_limiting_cycle(a, idstat, d_stat, v_stat, t):
    X, Y, Z = integrate(t, a, idstat, d_stat, v_stat, model=regular_attractor)
    fig = plt.figure(figsize=(12, 9))
    ax = fig.gca(projection='3d')
    ax.xaxis.set_pane_color((1, 1, 1, 1))
    ax.yaxis.set_pane_color((1, 1, 1, 1))
    ax.zaxis.set_pane_color((1, 1, 1, 1))
    ax.plot(X, Y, Z, color='r', alpha=0.7, linewidth=0.7)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()

def main():
    a_stat = [1.1062, 1.2971, 1.5013, 1.803, 1.88, 1.9]
    idstat = tuple((0.06, 1, 0.038))
    d_stat = [10, 10, 10, 20]
    w_stat = [1, 2, 0.405, 1]
    t = np.linspace(0, 1000, 10000)

    plot_animation_variablility()

    #for a in a_stat:
    #    plot_3d_limiting_cycle(a, idstat, d_stat, w_stat, t)
    pass

if __name__ == '__main__':
    main()