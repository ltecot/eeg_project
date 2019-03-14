import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python import keras as kt
import keras
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, RNN, SimpleRNN
from keras.utils import to_categorical
from variational_autoencoder import VAE

# get_ipython().run_line_magic('matplotlib', 'inline')
# plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
# plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'

X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")
person_train_valid = np.load("person_train_valid.npy")
X_train_valid = np.load("X_train_valid.npy")
y_train_valid = np.load("y_train_valid.npy")
person_test = np.load("person_test.npy")
print ('Training/Valid data shape: {}'.format(X_train_valid.shape))
print ('Test data shape: {}'.format(X_test.shape))
print ('Training/Valid target shape: {}'.format(y_train_valid.shape))
print ('Test target shape: {}'.format(y_test.shape))
print ('Person train/valid shape: {}'.format(person_train_valid.shape))
print ('Person test shape: {}'.format(person_test.shape))

# fomat data to (batch_size, timesteps, input_dim)
k_X_train = np.swapaxes(X_train_valid, 1, 2)
k_X_test = np.swapaxes(X_test, 1, 2)
print("Swapped axes:",k_X_train.shape, k_X_test.shape)
# remove VOG channels (23-25)
k_X_train = k_X_train[: , : , :22 ]
k_X_test = k_X_test[: , : , :22 ]
print("Removed VOG channels:", k_X_train.shape, k_X_test.shape)
# encode output labels
print("Raw labels:", y_train_valid[0:10])
k_y_train = y_train_valid - 769
k_y_test = y_test - 769
print("Fixed:", k_y_train[0:10])
k_y_train_categ = to_categorical(k_y_train, 4)
k_y_test_categ = to_categorical(k_y_test, 4)
print("Categorical one-hot encoding:\n",k_y_train_categ[0:3])

k_hidden_size = 200

vae = VAE(k_X_train.shape[2], 64, 3)
vae.load_weights("vae_mlp_eeg.h5")

def convert_data(x, embedding_dim=3):
    old_shape = x.shape[:2]
    print(old_shape)
    x = np.reshape(x, (-1, x.shape[2]))
    embedded_data = vae.forward(x)
    new_shape = (*old_shape, embedding_dim)
    x = np.reshape(embedded_data, new_shape)
    return x

use_embeddings = True
if use_embeddings:
    k_X_train = convert_data(k_X_train)
    k_X_test = convert_data(k_X_test)

 
model = Sequential()
model.add(SimpleRNN(k_hidden_size, input_shape=k_X_train.shape[1:]))
model.add(Dense(4, activation='softmax'))

model.compile(loss = 'categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'],)

model.summary()
model.fit(k_X_train, k_y_train_categ, epochs=10)

output = model.evaluate(k_X_test, k_y_test_categ)
print(output)
