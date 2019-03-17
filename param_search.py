import numpy as np
import matplotlib.pyplot as plt
import numpy as np
#set seed for reproducability
np.random.seed(1337)

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python import keras as kt
import keras
from sklearn.utils import shuffle
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, RNN, SimpleRNN, GRU
from keras.utils import to_categorical
import time
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.layers import Dropout
from keras import backend as be
from keras.callbacks import CSVLogger
import gc
from variational_autoencoder import transform_data_with_VAE, VAE


### NOW FOR training
def create_model(learn_rate=0.001, clip_value=1, cell_type='LSTM', num_units=100, dropout=False, add_conv=False, input_dim=None, num_filters=32, kernel_size=10, pool_size=5, stride_size=4):
    model = Sequential()
    if add_conv:
        model.add(keras.layers.Conv1D(num_filters, kernel_size, input_shape=input_dim, strides=stride_size))
        if dropout:
            model.add(Dropout(0.5))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv1D(num_filters, kernel_size, input_shape=input_dim, strides=stride_size))
        if dropout:
            model.add(Dropout(0.5))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPool1D(pool_size=pool_size))
    if cell_type == 'LSTM':
        model.add(keras.layers.CuDNNLSTM(num_units, input_shape=input_dim))
    elif cell_type == 'GRU':
        model.add(keras.layers.CuDNNGRU(num_units, input_shape=input_dim))

    if dropout:
        model.add(Dropout(0.5))
    model.add(keras.layers.BatchNormalization())
    model.add(Dense(4, activation="softmax"))


    optimizer = keras.optimizers.RMSprop(lr=learn_rate, clipvalue=1)
    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizer,
                  metrics=["accuracy"])

    return model

def crop_data_aug(X, y, crop):
    crop_size = crop # 3 # 100
    original_train_X = X  # test_array_x # p0_train_X
    original_train_y = y # test_array_y # p0_train_y

    N, T, C = original_train_X.shape
    print("Original Data:", original_train_X.shape, original_train_y.shape)
    cropped_train_X = np.zeros((N*(T-crop_size+1), C, crop_size))
    cropped_train_y = np.zeros(N*(T-crop_size+1))
    crops_per_sample = T-crop_size+1

    for n in np.arange(N):
        crop_count = 0
        for t in np.arange(T-crop_size+1):
            idx = n*crops_per_sample + crop_count
            cropped_train_X[idx] = original_train_X[n, :, t:t+crop_size]
            cropped_train_y[idx] = original_train_y[n]
            crop_count = crop_count + 1
            
    print("Cropped Data:", cropped_train_X.shape, cropped_train_y.shape)
    
    return cropped_train_X, cropped_train_y


def transform_data(X, y, crop=False):
    X = np.swapaxes(X, 1, 2)
    print("Swapped axes:", X.shape)
    X = X[: , : , :22 ]
    print("Removed VOG channels:", X.shape)
    # encode output labels
    print("Raw labels:", y[0:10])
    y = y- 769
    print("Fixed:", y[0:10])
    y = to_categorical(y, 4)
    print("Categorical one-hot encoding:\n",y[0:3])

    return (X, y)


if __name__ == '__main__':
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



    X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=0.2)

    X_train, y_train = transform_data(X_train, y_train)
    X_valid, y_valid = transform_data(X_valid, y_valid)
    X_test, y_test= transform_data(X_test, y_test)

    use_vae=False
    if use_vae:
        print("Tranforming data using VAE")
        vae = VAE(X_train.shape[2], 64, 3)
        vae.load_weights("vae_mlp_eeg.h5")

        X_train = transform_data_with_VAE(vae, X_train)
        X_valid = transform_data_with_VAE(vae, X_valid)
        X_test = transform_data_with_VAE(vae, X_test)

    X_train, y_train= shuffle(np.concatenate((X_train, X_train, X_train, X_train, X_train, X_train)), np.concatenate((y_train, y_train, y_train, y_train, y_train, y_train)))

    input_dim = X_train.shape[1:]

    lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1)
    csv_logger = CSVLogger('training.log')


    print("DATA READY FOR TRAINING!!!!")
    start = time.time()
    model = create_model(learn_rate=0.001, cell_type='GRU', num_units=64, dropout=True, add_conv=True, input_dim=input_dim, num_filters=64, pool_size=8, kernel_size=32, stride_size=4)
    history = model.fit(X_train, y_train, epochs=250, batch_size=64, validation_data=(X_valid, y_valid), verbose=1, callbacks=[lr_scheduler, csv_logger, early_stopping])
    print("Trained in {}".format(time.time()-start))
    print("val_acc max: {:.3f}  mean: {:.3f}".format(max(history.history['val_acc']), sum(history.history['val_acc']) / len(history.history['val_acc'])))
    print("\nTEST SET accuracy:")
    print(model.evaluate(X_test, y_test))
    model.save("best.h5")

    # lrs = [0.01, 0.03, 0.001, 0.003, 0.001]#, 0.0003, 0.0001]
    # lrs = [0.03]
    # num_hidden_dim = [64]
    # cell_types = ['LSTM']
    # use_dropout = [True]
    # pool_sizes = [4,8, 16]
    # kernel_sizes =[8, 16, 32]
    # stride_sizes = [2, 4]
    # filter_sizes = [32, 64]

    # results = dict()
    # for lr in lrs:
        # for h_dim in num_hidden_dim:
            # for c_t in cell_types:
                # for d in use_dropout:
                    # for pool_size in pool_sizes:
                        # for kernel_size in kernel_sizes:
                            # for num_filters in filter_sizes:
                                # for ss in stride_sizes:
                                    # key = (lr, h_dim, c_t, d, True, pool_size, kernel_size, num_filters, ss)
                                    # try:
                                        # start = time.time()
                                        # model = create_model(learn_rate=lr, cell_type=c_t, num_units = h_dim, dropout=d, add_conv = True, input_dim=input_dim, num_filters=num_filters, pool_size=pool_size, kernel_size=kernel_size, stride_size=ss)
                                        # history = model.fit(X_train, y_train, epochs=15, batch_size=128, validation_data=(X_valid, y_valid), verbose=0)
                                        # print("Trained {} in {}".format(key, time.time()-start))
                                        # print("val_acc: ", history.history['val_acc'][-1])
                                        # results[key] = history.history
                                    # except:
                                        # results[key] = "EXCEPTION"

    # import pickle
    # f = open("dict.pkl", "wb+")
    # pickle.dump(results, f)
    # f.close()
