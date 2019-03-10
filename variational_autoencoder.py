# This model is copied and modified from the Keras VAE example. See link below.
# https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py

'''Example of VAE on MNIST dataset using MLP
The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to generate latent vectors.
The decoder can be used to generate MNIST digits by sampling the
latent vector from a Gaussian distribution with mean = 0 and std = 1.
# Reference
[1] Kingma, Diederik P., and Max Welling.
"Auto-Encoding Variational Bayes."
https://arxiv.org/abs/1312.6114
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

from scipy import integrate

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation

# Based off code from following link
# https://jakevdp.github.io/blog/2013/02/16/animating-the-lorentz-system-in-3d/
def plot_3D(model, x_t, y_t, num_show=10, plot=False):

    t = np.arange(0, 1000)
    rand_indexes = np.random.choice(x_t.shape[0], num_show, replace=False)
    # print(rand_indexes)
    x_t = x_t[rand_indexes, :, :]
    y_t = y_t[rand_indexes]

    # print(x_t.shape)

    # Set up figure & 3D axis for animation
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    ax.axis('on')

    # choose a different color for each trajectory
    # colors = plt.cm.jet(np.linspace(0, 1, x_t.shape[0]))
    # Different color for each action class.
    colors = plt.cm.jet(y_t / float(np.amax(y_t)))
    # print(colors)

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
        # we'll step two time-steps per frame.  This leads to nice results.
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
    anim.save('vae_viz.mp4', fps=15, extra_args=['-vcodec', 'libx264'])

    if plot:
        plt.show()

# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


class VAE(object):

    def __init__(self, original_dim, intermediate_dim, latent_dim):
        # network parameters
        input_shape = (original_dim, )

        # VAE model = encoder + decoder
        # build encoder model
        inputs = Input(shape=input_shape, name='encoder_input')
        x = Dense(intermediate_dim, activation='relu')(inputs)
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

        # instantiate encoder model
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        # encoder.summary()
        plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

        # build decoder model
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        x = Dense(intermediate_dim, activation='relu')(latent_inputs)
        outputs = Dense(original_dim, activation='sigmoid')(x)

        # instantiate decoder model
        decoder = Model(latent_inputs, outputs, name='decoder')
        # decoder.summary()
        plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

        # instantiate VAE model
        outputs = decoder(encoder(inputs)[2])
        vae = Model(inputs, outputs, name='vae_mlp')

        self.vae = vae
        self.encoder = encoder
        self.decoder = decoder
        self.inputs = inputs
        self.outputs = outputs
        self.z_mean = z_mean
        self.z_log_var = z_log_var
        self.original_dim = original_dim
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim

    def train(self, x_train, x_test, batch_size=128, epochs=1, use_mse=True):

        x_train = np.random.permutation(x_train)
        inputs = self.inputs
        outputs = self.outputs
        original_dim = self.original_dim
        z_mean = self.z_mean
        z_log_var = self.z_log_var
        vae = self.vae

        # VAE loss = mse_loss or xent_loss + kl_loss
        if use_mse:
            reconstruction_loss = mse(inputs, outputs)
        else:
            reconstruction_loss = binary_crossentropy(inputs,
                                                      outputs)

        reconstruction_loss *= original_dim
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)
        vae.compile(optimizer='adam')
        # vae.summary()
        plot_model(vae,
                   to_file='vae_mlp.png',
                   show_shapes=True)

        # train the autoencoder
        vae.fit(x_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, None))
        vae.save_weights('vae_mlp_eeg.h5')

    def forward(self, x, batch_size=128):
        z_mean, _, _ = self.encoder.predict(x, batch_size=batch_size)
        return z_mean

    def load_weights(self, weights):
        self.vae.load_weights(weights)

if __name__ == '__main__':

    # weights = None
    weights = "vae_mlp_eeg.h5"

    def format_data(x, y):
        x = np.swapaxes(x, 1, 2)
        x = x[: , : , :22 ]
        x = np.reshape(x, (-1, x.shape[2]))
        y = y-769
        y = np.repeat(y, 1000, axis=0)
        return x, y

    X_test = np.load("X_test.npy")
    y_test = np.load("y_test.npy")
    original_y_test = y_test-769
    person_train_valid = np.load("person_train_valid.npy")
    X_train_valid = np.load("X_train_valid.npy")
    y_train_valid = np.load("y_train_valid.npy")
    person_test = np.load("person_test.npy")

    original_x_test = X_test
    X_test, y_test = format_data(X_test, y_test)
    X_train_valid, y_train_valid = format_data(X_train_valid, y_train_valid)

    # print(X_test.shape)
    # print(y_test.shape)
    # print(X_train_valid.shape)
    # print(y_train_valid.shape)

    vae = VAE(X_train_valid.shape[1], 100, 3)
    if weights:
        vae.load_weights(weights)
    else:
        vae.train(X_train_valid, X_test, batch_size=128, epochs=1, use_mse=True)

    test_out = vae.forward(X_test)
    test_out = np.reshape(test_out, (-1, 1000, 3))
    # print(test_out.shape)
    # print(test_out)
    plot_3D(vae, test_out, original_y_test, plot=True)

