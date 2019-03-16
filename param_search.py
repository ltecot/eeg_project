import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python import keras as kt
import keras
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, RNN, SimpleRNN, GRU
from keras.utils import to_categorical
import time
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.layers import Dropout
from keras import backend as be
import gc


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

idx_p0_train = np.where(person_train_valid == 0)[0]
idx_p0_test = np.where(person_test == 0)[0]

p0_train_X = X_train_valid[idx_p0_train] 
p0_train_y = y_train_valid[idx_p0_train]

p0_test_X = X_test[idx_p0_test]
p0_test_y = y_test[idx_p0_test]

print(p0_train_X.shape)
print(p0_train_y.shape)
print(p0_test_X.shape)
print(p0_test_y.shape)

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
cropped_train_X, cropped_train_y = crop_data_aug(data_train_X, data_train_y, 300)
cropped_val_X, cropped_val_y = crop_data_aug(data_val_X, data_val_y, 300)

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

input_dim = k_X_train.shape[1:]

print("DATA READY FOR TRAINING!!!!")
### NOW FOR training
def create_model(learn_rate=0.001, clip_value=1, cell_type='LSTM', num_units=100, dropout=False, add_conv=False):


    model = Sequential()

    if add_conv:
        model.add(keras.layers.Conv1D(22, 10, input_shape=input_dim))
        if dropout:
            model.add(Dropout(0.5))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv1D(22, 10))
        if dropout:
            model.add(Dropout(0.5))
        model.add(keras.layers.BatchNormalization())

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

lrs = [0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]
num_hidden_dim = [32, 64, 128]
cell_types = ['LSTM', 'GRU']
use_dropout = [False, True]

# model = KerasClassifier(build_fn=create_model, epochs=5, batch_size=256)

# param_grid = dict(learn_rate=lrs, num_units=num_hidden_dim, cell_type = cell_types, dropout=use_dropout)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, pre_dispatch=1, refit=False)
# grid_result = grid.fit(k_X_train, k_y_train_categ)

# # summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
        # print("%f (%f) with: %r" % (mean, stdev, param))
results = dict()
for lr in lrs:
    for h_dim in num_hidden_dim:
        for c_t in cell_types:
            for d in use_dropout:
                start = time.time()
                key = (lr, h_dim, c_t, d, True)
                model = create_model(learn_rate=lr, cell_type=c_t, num_units = h_dim, dropout=d, add_conv = True)
                history = model.fit(k_X_train, k_y_train_categ, epochs=5, batch_size=256, validation_split=0.1, verbose=0)
                results[key] = history.history

                print("Trained {} in {}".format(key, time.time()-start))
                print("val_acc: ", history.history['val_acc'][-1])


import pickle
f = open("dict.pkl", "wb+")
pickle.dump(results, f)
f.close()

                # print(test_model(learn_rate=lr, cell_type=c_t, num_units = h_dim, dropout=d, add_conv = True))
