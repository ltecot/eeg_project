# This model is a modified version of the Keras VAE example. See link below.
# https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py

# Reference
# [1] Kingma, Diederik P., and Max Welling.
# "Auto-Encoding Variational Bayes."
# https://arxiv.org/abs/1312.6114

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Because OSX complains about matplotlib
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")

from keras.layers import Lambda, Input, Dense
from keras.models import Model, load_model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

from plotting_utils import plot_3D

from sklearn.manifold import TSNE

# Utils for translating data 
def transform_data_with_VAE(vae_model, data):
    """ Transform data with VAE. Input the VAE model
        and the data. Can be any dimension as long as 
        last dimension matches up with input dimension of VAE.
    """
    original_shape = data.shape
    reshaped_data = np.reshape(data, (-1, original_shape[-1]))
    out = vae_model.forward(reshaped_data)
    new_shape = list(original_shape)
    new_shape[-1] = out.shape[-1]
    new_shape = tuple(new_shape)
    return np.reshape(out, new_shape)

def transform_data_with_TSNE(latent_dim, data):
    """ Transform data with t-sne. Last dim will be 
        transformed to the latent_dim.
    """
    original_shape = data.shape
    reshaped_data = np.reshape(data, (-1, original_shape[-1]))
    out = TSNE(n_components=3, verbose=1).fit_transform(reshaped_data)
    new_shape = list(original_shape)
    new_shape[-1] = out.shape[-1]
    new_shape = tuple(new_shape)
    return np.reshape(out, new_shape)

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
    """ Class to use and train a VAE
    Args:
        original_dim: Last dimension of the input data. (22 for EEG)
        intermediate_dim: Dimension for all hidden layers in encoder/decoder.
        latent_dim: Dimension of latent state. (3 for 3D data)
    """
    def __init__(self, original_dim, intermediate_dim, latent_dim):

        # network parameters
        input_shape = (original_dim, )

        # VAE model = encoder + decoder
        # build encoder model
        inputs = Input(shape=input_shape, name='encoder_input')
        h1 = Dense(intermediate_dim, activation='relu')(inputs)
        h2 = Dense(intermediate_dim, activation='relu')(h1)
        x = Dense(intermediate_dim, activation='relu')(h2)
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
        h2 = Dense(intermediate_dim, activation='relu')(x)
        h1 = Dense(intermediate_dim, activation='relu')(h2)
        outputs = Dense(original_dim, activation='linear')(h1)

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

    def train(self, x_train, x_valid, batch_size=128, epochs=1, use_mse=True, save_name="vae_mlp_eeg"):
        """ Trains the VAE
        Args:
            x_train: Data to use for training. Must be 2D, and last dim must match original_dim.
                     For EEG training, we flatten out the time steps of all the trails for this.
            x_valid: Data for validation during training. Same dimension properties as x_train.
            batch_size: Batch size
            epochs: number of epochs to train
            use_mse: Whether to use MSE or cross entropy in training. Generally should stay true.
            latent_dim: Dimension of latent state. (3 for 3D data)
        """

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
                validation_data=(x_valid, None))
        vae.save(save_name + '.h5')
        vae.save_weights(save_name + '_weights.h5')

    def load_weights(self, weights):
        """ Load Weights. Intended to be used if you don't want to train again.
        Args:
            weights: Name of the weights file. Can be a relative or absolute file path.
        """
        self.vae.load_weights(weights)

    def forward(self, x, batch_size=128):
        """ Encode the data into the latent state.
        Args:
            x_train: Data to encode. Must be 2D, and last dim must match original_dim.
            batch_size: Batch size
        Returns:
            2D array of encodings. Will be same dimensionality as input, except the 
            last dimension will be converted to the latent_dim size.
        """
        z_mean, _, _ = self.encoder.predict(x, batch_size=batch_size)
        return z_mean

    def reconstruct(self, x, batch_size=128):
        ''' Re-create x data. For testing purposes
        '''
        z_mean, _, _ = self.encoder.predict(x, batch_size=batch_size)
        xp = self.decoder.predict(z_mean, batch_size=batch_size)
        return xp

if __name__ == '__main__':

    model_name = "vae_mlp_eeg"
    train_model = False

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

    # Grab one person
    # idx_p0_train = np.where(person_train_valid == 0)[0]
    # idx_p0_test = np.where(person_test == 0)[0]

    # X_train_valid = X_train_valid[idx_p0_train] 
    # y_train_valid = y_train_valid[idx_p0_train]

    # X_test = X_test[idx_p0_test]
    # y_test = y_test[idx_p0_test]
    # original_y_test = y_test-769

    # Split train and valid. 90-10 split
    X_train_valid_permute = np.random.permutation(X_train_valid)
    split_ind = int(9 * X_train_valid.shape[0] / 10)
    X_train, X_valid = X_train_valid[:split_ind], X_train_valid[split_ind:]
    y_train, y_valid = y_train_valid[:split_ind], y_train_valid[split_ind:]

    original_x_test = X_test
    X_test, y_test = format_data(X_test, y_test)
    X_train, y_train = format_data(X_train, y_train)
    X_valid, y_valid = format_data(X_valid, y_valid)

    vae = VAE(X_train.shape[1], 64, 3)
    if not train_model:
        # vae = load_model(model_name + '.h5')
        vae.load_weights(model_name + '_weights.h5')
    else:
        vae.train(X_train, X_valid, batch_size=128, epochs=5, use_mse=True, save_name=model_name)

    inds_0 = np.where(original_y_test==0)
    inds_1 = np.where(original_y_test==1)
    inds_2 = np.where(original_y_test==2)
    inds_3 = np.where(original_y_test==3)
    # rand_indexes = np.concatenate((inds_0[0][:2], inds_1[0][:2], inds_2[0][:2], inds_3[0][:2]))
    rand_indexes = np.concatenate((np.random.choice(inds_0[0], 2, replace=False),
                                np.random.choice(inds_1[0], 2, replace=False),
                                np.random.choice(inds_2[0], 2, replace=False),
                                np.random.choice(inds_3[0], 2, replace=False)))
    plot_y_test = original_y_test[rand_indexes]

    # Plot test data, VAE
    X_test_small = np.reshape(X_test, (-1, 1000, 22))
    X_test_small = X_test_small[rand_indexes, :, :]
    X_test_latent_vae = transform_data_with_VAE(vae, X_test_small)
    plot_3D(X_test_latent_vae, plot_y_test, plot=True, file_name='vae_viz2.mp4')

    # Plot test TSNE
    # X_test_latent_tsne = transform_data_with_TSNE(3, X_test_small)
    # plot_3D(X_test_latent_tsne, plot_y_test, plot=True, file_name='tsne_viz2.mp4')

    # 1903000/1903000 [==============================] - 323s 170us/step - loss: 145.9841 - val_loss: 152.7765
    # Epoch 2/5
    # 1903000/1903000 [==============================] - 371s 195us/step - loss: 136.1010 - val_loss: 151.9541
    # Epoch 3/5
    # 1903000/1903000 [==============================] - 382s 201us/step - loss: 131.5169 - val_loss: 144.1513
    # Epoch 4/5
    # 1903000/1903000 [==============================] - 359s 189us/step - loss: 131.0245 - val_loss: 149.1800
    # Epoch 5/5
    # 1903000/1903000 [==============================] - 348s 183us/step - loss: 131.2396 - val_loss: 150.1427
    print("valid average difference", np.average(np.abs(X_valid - vae.reconstruct(X_valid))))
    print("valid average value", np.average(np.abs(X_valid)))
    print("test average difference", np.average(np.abs(X_test - vae.reconstruct(X_test))))
    print("test average value", np.average(np.abs(X_test)))

