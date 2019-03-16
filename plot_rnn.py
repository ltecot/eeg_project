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

