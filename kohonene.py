# python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
# from mpl_toolkits.mplot3d import Axes3D

class kohonen:
    def __init__(self):
        self.timeElapsed = 0
        self.weight = np.array([])
        self.initWeight = self.weight.copy()
        # self.dim = self.initWeight.shape
        self.alpha = 0.01

    def setWeight(self, n=100, m=2):
        """ Setting Initial Weights(Matrix), default = 100 * 2"""
        self.weight = np.random.random((n, m))

    def step(self, inputVector, loop):
        """step per loop"""
        assert self.weight.size != 0, "[ - ] weight is not been set. Use \".setWeight(n=100, m=2)\" first."
        assert inputVector.shape[1] == self.weight.shape[1], "[ - ] Vectors are not same dimension."

        subLoop = inputVector.shape[0] - 1
        count = 0
        while count < loop:
            # /------- get distance ------
            dist = []
            for i in range(0, subLoop):
                k = np.array(((self.weight[..., 0] - inputVector[i, 0]) ** 2) + ((self.weight[..., 1] - inputVector[i,1]) ** 2))
                dist.append(k)

            # /------- extract minimum vector index ------
            index = []
            for i in range(0, subLoop):
                l = dist[i].argmin()
                index.append(l)

            for i in range(0, subLoop):
                self.weight[index[i]] -= self.alpha * (self.weight[index[i]] - inputVector[i])

            # /------- counting loop ------# /
            count += 1

colors = ["red"] * 50
colors.extend(["green"] * 50)
colors.extend(["blue"] * 50)
markers = ['o']
markers.extend(['x'])
markers.extend(['^'])

# fig, ax = plt.subplots()
# axes = [ax]

# /------- extract minimum vector index ------






def overlayData(data):
    x = data[..., 0]
    y = data[..., 1]
    # l = int(data.shape[1] - 1)
    # for i in range(0,l):
        # z = data[..., 2]
        # fig = plt.figure()
        # ax = fig.
    plt.scatter(x, y, marker=markers, color=colors)

# /------- initiating ------
np.random.seed(0)
inputVector = np.random.random((1000, 2))

print(inputVector[1:5])

iteration = 100

layer = kohonen()
layer.setWeight(100,2)

print(layer.initWeight[1:5])

layer.step(inputVector, iteration)
# overlayData(layer.weight)

print(layer.weight[1:5])

plt.show()



"""
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
    # initialize animation
    global box, rect
    particles.set_data([], [])
    rect.set_edgecolor('none')
    return particles, rect


def animate(i):
    # perform animation step
    global box, rect, dt, ax, fig
    box.step(dt)

    ms = int(fig.dpi * 2 * box.size * fig.get_figwidth()
             / np.diff(ax.get_xbound())[0])

    # update pieces of the animation
    rect.set_edgecolor('k')
    particles.set_data(box.state[:, 0], box.state[:, 1])
    particles.set_markersize(ms)
    return particles, rect


anim = animation.FuncAnimation(fig, animate, init_func=init, frames=loop, interval=1000, blit=True)
plt.show()
"""