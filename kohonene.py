# python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# import time
# from matplotlib import animation
# from mpl_toolkits.mplot3d import Axes3D

class kohonen:
    def __init__(self):
        self.weight = np.array([])
        self.initWeight = np.array([])
        # self.dim = self.initWeight.shape
        self.alpha = 0.01

    def setWeight(self, n=100, m=2):
        """ Setting Initial Weights(Matrix), default = 100 * 2"""
        self.weight = np.random.random((n, m))
        self.initWeight = self.weight.copy()

    def step(self, inputVector, loop, storeImg=False):
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
            if storeImg == True:
                overlayData(layer.weight, c='red', a=(count / iteration))
                plt.savefig(str(count) + "_figure.png")

# fig, ax = plt.subplots()
# axes = [ax]


def overlayData(data, c='red', a=0.5):
    x = data[..., 0]
    y = data[..., 1]
    z = data[..., 2]
    # l = int(data.shape[1] - 1)
    # for i in range(0,l):
        # z = data[..., 2]
        # fig = plt.figure()
        # ax = fig.
 #   plt.scatter(x, y, marker='o', color=c, alpha=a)
    ax.scatter(x, y, z, c=c, alpha=a, marker='o')



# /------- initiating ------
np.random.seed(0)
dimension = 3
iteration = 100
testset = 100
inputVector = np.random.random((testset, dimension))
# overlayData(inputVector, 'black', 0.75,storeImg=False)
# print(inputVector[1:5], inputVector.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')



layer = kohonen()
layer.setWeight(100,dimension)
overlayData(layer.initWeight, 'black', 0.3)
# print(layer.initWeight[1:5], layer.initWeight.shape)
layer.step(inputVector, iteration, storeImg=True)
overlayData(layer.weight, 'black', 0.9)
# print(layer.weight[1:5], layer.weight.shape)

# /------- draw animation and background ------
# fig = plt.figure()
# fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
# ax = fig.add_subplot(111)

# `particles` holds the location of the particles
# particles, = ax.plot([], [], 'bo', ms=6)
#
# # `rect` is the boarders
# rect = plt.Rectangle(
#     (0,0),                      # start point. right low
#     2,                          # width
#     2                           # height
# )
#
# ax.add_patch(rect)

def init():
    global layer, rect
    particles.set_data([],[])
    rect.set_edgecolor('none')
    return particles, rect

def animate(i):
    global layer, rect, dt, ax, fig
    layer.step(inputVector, iteration)
    # ms = int(fig.dpi * 2 * layer.shape[0] * fig.get_figwidth() / np.diff(ax.get_xbound())[0])
    rect.set_edgecolor('k')
    particles.set_data(layer.weight[..., 0],layer.weight[..., 1])
    # particles.set_makersize(5)
    print(layer.weight[..., 0], layer.weight[..., 1])
    return particles, rect

# ani = animation.FuncAnimation(fig, animate, frames=iteration, interval=100, blit=True, init_func=init)
# ani.save('particle_box.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
print("start encoding.")
# plt.show()
