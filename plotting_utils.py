
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Because OSX complains about matplotlib
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation

# Based off code from following link
# https://jakevdp.github.io/blog/2013/02/16/animating-the-lorentz-system-in-3d/
def plot_3D(x_t, y_t, num_show=None, plot=False, file_name=None):
    """ Plots data in 3D over time.
        Note that this generally doesn't work in juypter notebooks.
    Args:
        x_t: Data to plot. Should be 3 dimensions:
             First Dim: Each trail.
             Second Dim: Each time point of the trail
             Third Dim: Each dimension of the data point. Should be of size 3.
        y_t: Label of the plotting data. Only used to color code the plot lines.
             1D array with same size as the first dimension of x_t.
        num_show: If not None, will randomly select this number of trails to plot instead
                  of the whole data set.
        plot: Whether to plot.
        file_name: File name to save the plot video to.
    """

    t = np.arange(0, x_t.shape[1])
    if num_show:
        rand_indexes = np.random.choice(x_t.shape[0], num_show, replace=False)
        x_t = x_t[rand_indexes, :, :]
        y_t = y_t[rand_indexes]

    # Set up figure & 3D axis for animation
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    ax.axis('on')

    # Different color for each action class.
    colors = plt.cm.jet((y_t - np.amin(y_t)) / float(np.amax(y_t)))

    # set up lines and points
    lines = sum([ax.plot([], [], [], '-', c=c)
                 for c in colors], [])
    pts = sum([ax.plot([], [], [], 'o', c=c)
               for c in colors], [])

    # prepare the axes limits
    ax.set_xlim((-2, 2))
    ax.set_ylim((-2, 2))
    ax.set_zlim((-2, 2))

    # set point-of-view: specified by (altitude degrees, azimuth degrees)
    ax.view_init(30, 0)

    # initialization function: plot the background of each frame
    def init():
        for line, pt in zip(lines, pts):
            line.set_data([], [])
            line.set_3d_properties([])

            pt.set_data([], [])
            pt.set_3d_properties([])
        return lines + pts

    # animation function.  This will be called sequentially with the frame number
    def animate(i):
        i = (i) % x_t.shape[1]
        for line, pt, xi in zip(lines, pts, x_t):
            x, y, z = xi[:i].T
            line.set_data(x, y)
            line.set_3d_properties(z)
            pt.set_data(x[-1:], y[-1:])
            pt.set_3d_properties(z[-1:])
        ax.view_init(30, 0.3 * i)
        fig.canvas.draw()
        return lines + pts

    # instantiate the animator.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=500, interval=30, blit=True)

    # Save as mp4. This requires mplayer or ffmpeg to be installed
    if file_name:
        anim.save(file_name, fps=15, extra_args=['-vcodec', 'libx264'])

    if plot:
        plt.show()