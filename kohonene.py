# python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
# from mpl_toolkits.mplot3d import Axes3D

# def showData(data):
#     x = data[..., 0]
#     y = data[..., 1]
#     z = data[..., 2]
#     fig = plt.figure()
#     ax = Axes3D(fig)
#     ax.scatter(x, y, z, c=colors)
#     plt.show()


colors = ["red"] * 50
colors.extend(["green"] * 50)
colors.extend(["blue"] * 50)


class kohonen(data):
    def _init__(self, initState=data):
        self.timeElapsed = 0
        self.initState = np.array(initState, dtype=float)
        self.state = initState.copy()
        self.alpha = 0.01
        self.dim = self.initState.shape

    def step(self, dt):
        """step per loop"""

        self.assignVec = dt
        temp =
        assignSet =
        self.stateSet = stateSet +


    def getDistance(self, p0, p1):
        p = (po - p1) ** 2


# set up initial state
np.random.seed(0)
inputData = np.random.random((10, 2))

box = kohonen(inputData)

initialWeight = np.random.random((100, 2))
alpha = 0.55
iteration = 100
loop = 0
td = array([[], [], []])
td[0] = initialWeight.copy()

while loop < iteration:
    for i in data:
        td = (td - i) ** 2
        dist = np.sqrt(td[..., 0] + td[..., 1])
        td[dist.argmin()] = td[dist.argmin()] + alpha * (i - td[dist.argmin()])
        # print(loop, dist.argmin(), td[dist.argmin()])
    loop += 1

fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-1, 1), ylim=(-1, 1))

particles, = ax.plot([], [], 'bo', ms=6)

rect = plt.Rectangle(box.bounds[::2],
                     box.bounds[1] - box.bounds[0],
                     box.bounds[3] - box.bounds[2],
                     ec='none', lw=2, fc='none')
ax.add_patch(rect)


def init():
    """initialize animation"""
    global box, rect
    particles.set_data([], [])
    rect.set_edgecolor('none')
    return particles, rect


def animate(i):
    """perform animation step"""
    global box, rect, dt, ax, fig
    box.step(dt)

    ms = int(fig.dpi * 2 * box.size * fig.get_figwidth()
             / np.diff(ax.get_xbound())[0])

    # update pieces of the animation
    rect.set_edgecolor('k')
    particles.set_data(box.state[:, 0], box.state[:, 1])
    particles.set_markersize(ms)
    return particles, rect


anim = animation.FuncAnimation(fig, animate, init_func=init, frames=10, interval=500, blit=True)
plt.show()
