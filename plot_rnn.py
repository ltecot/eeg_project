# Because OSX complains about matplotlib
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python import keras as kt
import keras
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, RNN, SimpleRNN, GRU
from keras.utils import to_categorical
from sklearn.manifold import TSNE
from plotting_utils import plot_3D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.layers import Dropout
import h5py

from param_search import transform_data
from variational_autoencoder import transform_data_with_TSNE

# Modified from param_search file.
def create_model_plot(learn_rate=0.001, clip_value=1, cell_type='LSTM', num_units=100, dropout=False, add_conv=False, input_dim=None, num_filters=32, kernel_size=10, pool_size=5, stride_size=4):
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
        # model.add(keras.layers.CuDNNLSTM(num_units, input_shape=input_dim))
        model.add(keras.layers.LSTM(num_units, input_shape=input_dim, return_sequences=True, name='cu_dnnlstm_1'))
    elif cell_type == 'GRU':
        # model.add(keras.layers.CuDNNGRU(num_units, input_shape=input_dim))
        model.add(keras.layers.GRU(num_units, input_shape=input_dim, return_sequences=True, name='cu_dnngru_1', reset_after=True))

    # if dropout:
    #     model.add(Dropout(0.5))
    # model.add(keras.layers.BatchNormalization())
    # model.add(Dense(4, activation="softmax"))


    # optimizer = keras.optimizers.RMSprop(lr=learn_rate, clipvalue=1)
    # model.compile(loss="categorical_crossentropy",
    #               optimizer=optimizer,
    #               metrics=["accuracy"])

    return model

def plot_GRU1_128units_dropout_subsampling100_valacc38():
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

    # Separate data by subject

    # samples from subject 0
    idx_p0_train = np.where(person_train_valid == 0)[0]
    idx_p0_test = np.where(person_test == 0)[0]

    p0_train_X = X_train_valid[idx_p0_train] 
    p0_train_y = y_train_valid[idx_p0_train]

    p0_test_X = X_test[idx_p0_test]
    p0_test_y = y_test[idx_p0_test]

    ################################
    # Sliding window subsampling

    def crop_data_aug(X, y, crop):
        crop_size = crop # 3 # 100
        original_train_X = X  # test_array_x # p0_train_X
        original_train_y = y # test_array_y # p0_train_y

        N, C, T = original_train_X.shape
        print("Original Data:", original_train_X.shape, original_train_y.shape)
        #print("X", original_train_X[300:305], "Y", original_train_y[300:305])
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

    # separate the train and val test sets before actually cropping 
    idx_val = np.arange(p0_train_X.shape[0])
    np.random.shuffle(idx_val)
    val_fraction = int(0.2*idx_val.shape[0])

    # separated train and test splits
    data_val_X = p0_train_X[0:val_fraction]
    data_val_y = p0_train_y[0:val_fraction]
    data_train_X = p0_train_X[val_fraction:]
    data_train_y = p0_train_y[val_fraction:]

    # augment using cropping
    cropped_train_X, cropped_train_y = crop_data_aug(data_train_X, data_train_y, 100)
    cropped_val_X, cropped_val_y = crop_data_aug(data_val_X, data_val_y, 100)


    ################################
    # rearrange data for Keras modules

    ##!!!!!! Now train and test data dimensions do not match

    # fomat data to (batch_size, timesteps, input_dim)
    k_X_train = np.swapaxes(cropped_train_X, 1, 2)
    k_X_val = np.swapaxes(cropped_val_X, 1, 2)
    k_X_test = np.swapaxes(p0_test_X, 1, 2)
    print("Swapped axes:",k_X_train.shape, k_X_test.shape)
    # remove VOG channels (23-25)
    k_X_train = k_X_train[: , : , :22 ]
    k_X_val = k_X_val[:, :, :22]
    k_X_test = k_X_test[: , : , :22 ]
    print("Removed VOG channels:", k_X_train.shape, k_X_test.shape)

    # encode output labels
    print("Raw labels:", cropped_train_y[0:10])
    k_y_train = cropped_train_y - 769
    k_y_val = cropped_val_y - 769
    k_y_test = p0_test_y - 769
    print("Fixed:", k_y_train[0:10])
    k_y_train_categ = to_categorical(k_y_train, 4)
    k_y_val_categ = to_categorical(k_y_val, 4)
    k_y_test_categ = to_categorical(k_y_test, 4)
    print("Categorical one-hot encoding:\n",k_y_train_categ[0:3])

    #################### RNN training
    input_dim = k_X_train.shape[1:]
    gru_units_sub = 128

    # Plot test data
    # rand_indexes = np.random.choice(k_X_val.shape[0], 8, replace=False)
    inds_0 = np.where(k_y_val==0)
    inds_1 = np.where(k_y_val==1)
    inds_2 = np.where(k_y_val==2)
    inds_3 = np.where(k_y_val==3)
    # rand_indexes = np.concatenate((inds_0[0][:2], inds_1[0][:2], inds_2[0][:2], inds_3[0][:2]))
    rand_indexes = np.concatenate((np.random.choice(inds_0[0], 2, replace=False),
                                np.random.choice(inds_1[0], 2, replace=False),
                                np.random.choice(inds_2[0], 2, replace=False),
                                np.random.choice(inds_3[0], 2, replace=False)))
    k_X_val_small = k_X_val[rand_indexes, :, :]
    k_y_val_small = k_y_val[rand_indexes]

    model = Sequential()
    model.add(GRU(gru_units_sub, recurrent_dropout=.4, input_shape=input_dim, return_sequences=True))
    saved_model = keras.models.load_model("GRU1-128units-dropout-subsampling100-valacc38.h5")
    saved_model.summary()
    gru_weights = model.get_layer("gru_1").get_weights()
    model.layers[-1].set_weights(gru_weights)
    model.summary()
    # model.layers.pop()
    # gru_weights = model.layers[-1].get_weights()
    # model.layers.pop()
    # model.summary()
    # model.add(GRU(gru_units_sub, recurrent_dropout=.4, input_shape=input_dim, return_sequences=True))
    # model.layers[-1].set_weights(gru_weights)
    # model.summary()

    print(k_X_val_small.shape)
    k_y_val_small_predict = model.predict(k_X_val_small)
    print(k_y_val_small_predict.shape)
    k_y_val_small_predict_embedded = TSNE(n_components=3, verbose=1).fit_transform(k_y_val_small_predict.reshape((-1, gru_units_sub))).reshape((8, 100, 3))
    print(k_y_val_small_predict_embedded.shape)

    plot_3D(k_y_val_small_predict_embedded, k_y_val_small, plot=True, file_name='gruval38_viz3.mp4')

# Trained on person 0
def plot_GRU_conv_65val(plot_conv_output=False):
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

    # Separate data by subject

    # samples from subject 0
    idx_p0_train = np.where(person_train_valid == 0)[0]
    idx_p0_test = np.where(person_test == 0)[0]

    p0_train_X = X_train_valid[idx_p0_train] 
    p0_train_y = y_train_valid[idx_p0_train]

    p0_test_X = X_test[idx_p0_test]
    p0_test_y = y_test[idx_p0_test]

    ################################
    # Sliding window subsampling

    def crop_data_aug(X, y, crop):
        crop_size = crop # 3 # 100
        original_train_X = X  # test_array_x # p0_train_X
        original_train_y = y # test_array_y # p0_train_y

        N, C, T = original_train_X.shape
        print("Original Data:", original_train_X.shape, original_train_y.shape)
        #print("X", original_train_X[300:305], "Y", original_train_y[300:305])
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

    # separate the train and val test sets before actually cropping 
    idx_val = np.arange(p0_train_X.shape[0])
    np.random.shuffle(idx_val)
    val_fraction = int(0.2*idx_val.shape[0])

    # separated train and test splits
    data_val_X = p0_train_X[0:val_fraction]
    data_val_y = p0_train_y[0:val_fraction]
    data_train_X = p0_train_X[val_fraction:]
    data_train_y = p0_train_y[val_fraction:]

    # augment using cropping
    cropped_train_X, cropped_train_y = crop_data_aug(data_train_X, data_train_y, 100)
    cropped_val_X, cropped_val_y = crop_data_aug(data_val_X, data_val_y, 100)


    ################################
    # rearrange data for Keras modules

    ##!!!!!! Now train and test data dimensions do not match

    # fomat data to (batch_size, timesteps, input_dim)
    k_X_train = np.swapaxes(cropped_train_X, 1, 2)
    k_X_val = np.swapaxes(cropped_val_X, 1, 2)
    k_X_test = np.swapaxes(p0_test_X, 1, 2)
    print("Swapped axes:",k_X_train.shape, k_X_test.shape)
    # remove VOG channels (23-25)
    k_X_train = k_X_train[: , : , :22 ]
    k_X_val = k_X_val[:, :, :22]
    k_X_test = k_X_test[: , : , :22 ]
    print("Removed VOG channels:", k_X_train.shape, k_X_test.shape)

    # encode output labels
    print("Raw labels:", cropped_train_y[0:10])
    k_y_train = cropped_train_y - 769
    k_y_val = cropped_val_y - 769
    k_y_test = p0_test_y - 769
    print("Fixed:", k_y_train[0:10])
    k_y_train_categ = to_categorical(k_y_train, 4)
    k_y_val_categ = to_categorical(k_y_val, 4)
    k_y_test_categ = to_categorical(k_y_test, 4)
    print("Categorical one-hot encoding:\n",k_y_train_categ[0:3])

    #################### RNN training
    # input_dim = k_X_train.shape[1:]
    # gru_units_sub = 128

    # Plot test data
    # rand_indexes = np.random.choice(k_X_val.shape[0], 8, replace=False)
    inds_0 = np.where(k_y_val==0)
    inds_1 = np.where(k_y_val==1)
    inds_2 = np.where(k_y_val==2)
    inds_3 = np.where(k_y_val==3)
    # rand_indexes = np.concatenate((inds_0[0][:2], inds_1[0][:2], inds_2[0][:2], inds_3[0][:2]))
    rand_indexes = np.concatenate((np.random.choice(inds_0[0], 2, replace=False),
                                np.random.choice(inds_1[0], 2, replace=False),
                                np.random.choice(inds_2[0], 2, replace=False),
                                np.random.choice(inds_3[0], 2, replace=False)))
    k_X_val_small = k_X_val[rand_indexes, :, :]
    k_y_val_small = k_y_val[rand_indexes]

    input_dim = k_X_train.shape[1:]
    gru_units_sub = 128

    model = Sequential()
    model.add(keras.layers.Conv1D(22, 10, input_shape=input_dim))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv1D(22, 10))
    model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.CuDNNGRU(gru_units_sub, input_shape=input_dim))
    model.add(keras.layers.GRU(gru_units_sub, input_shape=input_dim,
                    return_sequences=True,
                    reset_after=True,
                    name='cu_dnngru_1'))
    f = h5py.File('Conv-BN-Conv-BN-GRU1-128units-BN-subsampling300-valacc65-weights.h5', 'r')
    print(list(f.keys()))
    model.load_weights('Conv-BN-Conv-BN-GRU1-128units-BN-subsampling300-valacc65-weights.h5', by_name=True)
    model.summary()
    # model.layers.pop()
    # gru_weights = model.layers[-1].get_weights()
    # model.layers.pop()
    # model.summary()
    # model.add(GRU(gru_units_sub, recurrent_dropout=.4, input_shape=input_dim, return_sequences=True))
    # model.layers[-1].set_weights(gru_weights)
    # model.summary()

    # print(k_X_val_small.shape)
    # k_y_val_small_predict = model.predict(k_X_val_small)
    # print(k_y_val_small_predict.shape)
    # k_y_val_small_predict_embedded = TSNE(n_components=3, verbose=1).fit_transform(k_y_val_small_predict.reshape((-1, gru_units_sub))).reshape((8, 100, 3))
    # print(k_y_val_small_predict_embedded.shape)

    # plot_3D(k_y_val_small_predict_embedded, k_y_val_small, plot=True, file_name='gruval38_viz3.mp4')

    if plot_conv_output:
        layer_name = None
        intermediate_layer_model = Model(inputs=model.input,
                                        outputs=model.get_layer(layer_name).output)
        k_y_val_small_predict = intermediate_layer_model.predict(k_X_val_small)
        k_y_val_small_predict_embedded = transform_data_with_TSNE(3, k_y_val_small_predict)
        plot_3D(k_y_val_small_predict_embedded, k_y_val_small, plot=True, file_name='gruval38_viz3.mp4')
    else:  # Plot reccurent output
        k_y_val_small_predict = model.predict(k_X_val_small)
        k_y_val_small_predict_embedded = transform_data_with_TSNE(3, k_y_val_small_predict)
        plot_3D(k_y_val_small_predict_embedded, k_y_val_small, plot=True, file_name='gruval38_viz3.mp4')
    

def plot_best(plot_conv_output=False):
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

    # 
    X_train, y_train= shuffle(np.concatenate((X_train, X_train, X_train, X_train, X_train, X_train)), np.concatenate((y_train, y_train, y_train, y_train, y_train, y_train)))

    # X_train, y_train = crop_data_aug(X_train, y_train, 300)

    input_dim = X_train.shape[1:]

    # lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)
    # early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1)


    # print("DATA READY FOR TRAINING!!!!")
    # start = time.time()
    plot_model = create_model_plot(learn_rate=0.001, cell_type='GRU', num_units=64, dropout=True, add_conv=True, input_dim=input_dim, num_filters=64, pool_size=8, kernel_size=32, stride_size=4)
    # history = model.fit(X_train, y_train, epochs=250, batch_size=64, validation_data=(X_valid, y_valid), verbose=1, callbacks=[lr_scheduler, early_stopping])
    # print("Trained in {}".format(time.time()-start))
    # print("val_acc max: {:.3f}  mean: {:.3f}".format(max(history.history['val_acc']), sum(history.history['val_acc']) / len(history.history['val_acc'])))
    # print("\nTEST SET accuracy:")
    # print(model.evaluate(X_test, y_test))
    # model.save("best.h5")

    # original_model = keras.models.load_model("best.h5")
    # weights = original_model.layers.get_weights()
    # weights = weights[:-2]  # Get rid of last dense and batchnorm
    # plot_model.load_weights(weights)
    plot_model.load_weights('best_weights.h5', by_name=True)
    plot_model.summary()
    # f = h5py.File('best_weights.h5', 'r')
    # print(list(f.keys()))
    # new_weights = f['conv1d_1']['dropout_1']['batch_normalization_1']['conv1d_2']['dropout_2']['batch_normalization_2']['max_pooling1d_1']['cu_dnngru_1']
    # new_weights = [f['conv1d_1'],
    #                f['dropout_1'],
    #                f['batch_normalization_1'],
    #                f['conv1d_2'],
    #                f['dropout_2'],
    #                f['batch_normalization_2'],
    #                f['max_pooling1d_1'],
    #                f['cu_dnngru_1']]
    # plot_model.load_weights(new_weights)

    y_valid_original = np.argmax(y_valid, axis=1)
    inds_0 = np.where(y_valid_original==0)
    inds_1 = np.where(y_valid_original==1)
    inds_2 = np.where(y_valid_original==2)
    inds_3 = np.where(y_valid_original==3)
    # rand_indexes = np.concatenate((inds_0[0][:2], inds_1[0][:2], inds_2[0][:2], inds_3[0][:2]))
    rand_indexes = np.concatenate((np.random.choice(inds_0[0], 2, replace=False),
                                np.random.choice(inds_1[0], 2, replace=False),
                                np.random.choice(inds_2[0], 2, replace=False),
                                np.random.choice(inds_3[0], 2, replace=False)))
    X_val_small = X_valid[rand_indexes, :, :]
    y_val_small = y_valid_original[rand_indexes]
    
    if plot_conv_output:
        layer_name = None
        intermediate_layer_model = Model(inputs=plot_model.input,
                                        outputs=plot_model.get_layer(layer_name).output)
        y_valid_pred = intermediate_layer_model.predict(X_val_small)
        y_valid_pred_embedded = transform_data_with_TSNE(3, y_valid_pred)
        print(y_valid_pred_embedded.shape)
        plot_3D(y_valid_pred_embedded, y_val_small, plot=True, file_name='best_viz_conv.mp4')
    else:  # Plot reccurent output
        y_valid_pred = plot_model.predict(X_val_small)
        y_valid_pred_embedded = transform_data_with_TSNE(3, y_valid_pred)
        print(y_valid_pred_embedded.shape)
        plot_3D(y_valid_pred_embedded, y_val_small, plot=True, file_name='best_viz.mp4')

if __name__ == '__main__':
    # plot_GRU1_128units_dropout_subsampling100_valacc38()
    plot_GRU_conv_53val()
    # plot_best()