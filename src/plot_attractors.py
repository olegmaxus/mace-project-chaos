import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

"""
TODO: Rewrite the code in accordance
with standards, as it is quite messy

Acknowledgements:
Sicere thanks to Mr. Lutz Lehmann
who has provided us with the idea
and the code for plotting the strange
attractor for the given system of
differential equations

https://math.stackexchange.com/users/115115/lutz-lehmann
"""


def regular_system(t, v, a):
    x, y, z = v
    dx = x * (a - 0.06 * x - y / (x + 10))
    dy = y * (-1 + 2 * x / (x + 10) - 0.405 * z / (y + 10))
    dz = (z ** 2) * (0.038 - 1 / (y + 20 ))
    return dx, dy, dz


a = 1.803  # defining the paper-based comparison a-value

v0 = [29.9, 0.2, 15.0]  # deterministic initial condition

T = 40


def event(t, u, a):
    return u[1]-24


event.direction = -1


fig = plt.figure(figsize=(7, 4.5), constrained_layout=True)
gs = fig.add_gridspec(2, 3)
ax = [fig.add_subplot(gs[k, 2]) for k in [0, 1]]
ax3 = fig.add_subplot(gs[:, :2])
for k in range(10):
    v0 = np.random.uniform(size=3) * np.array([5, 10, 20]) + np.array([25, 0, 0])
    res = solve_ivp(regular_system, (0, 10 * T), v0, args=(a,), events=event, atol=1e-6, rtol=1e-9)
    res = solve_ivp(regular_system, (0, 10 * T), res.y_events[0][-1], args=(a,), events=event, atol=1e-6, rtol=1e-9)
    ax3.plot(res.y[0], res.y[2], lw=0.8)
    ax[0].plot(res.y_events[0][:, 0], res.y_events[0][:, 2], '-+', lw=0.5)
    ax[1].plot(res.y_events[0][:-1, 0], res.y_events[0][1:, 0], '-+', lw=0.5)
for ak in [ax3, *ax]:
    ak.grid()

ax3.set_xlabel("$x$")
ax3.set_ylabel("$z$")
ax[0].set_xlabel("$x_E$")
ax[0].set_ylabel("$z_E$")
ax[1].set_xlabel("$x_E[k]$")
ax[1].set_ylabel("$x_E[k+1]$")
fig.suptitle(f'behaviour of chaotic attractor at $a={a}$')
plt.show()

fig = plt.figure(figsize=(12, 9))
ax = fig.gca(projection='3d')
ax.xaxis.set_pane_color((1, 1, 1, 1))
ax.yaxis.set_pane_color((1, 1, 1, 1))
ax.zaxis.set_pane_color((1, 1, 1, 1))

for k in range(10):
    v0 = np.random.uniform(size=3) * np.array([5, 10, 20]) + np.array([25, 0, 0])
    res = solve_ivp(regular_system, (0, 10 * T), v0, args=(a,), events=event, atol=1e-6, rtol=1e-9)
    res = solve_ivp(regular_system, (0, 10 * T), res.y_events[0][-1], args=(a,), events=event, atol=1e-6, rtol=1e-9)
    ax.plot(res.y[0], res.y[1], res.y[2], alpha=0.7, linewidth=0.7)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title(f'Deterministic system with regular and chaotic attractor for $a={a}$')
plt.show()
