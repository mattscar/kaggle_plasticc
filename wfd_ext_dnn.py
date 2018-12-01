import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import interpolate
from scipy.stats import kurtosis

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

np.random.seed(3)
import keras
from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, Dense, Flatten, Dropout, Input, Concatenate

# Set constants
num_passbands = 6
curve_width = 1024
freq_width = curve_width//2
curve_time_duration = 857.6
curve_time_interval = curve_time_duration/float(curve_width)
num_splits = 10

# Hyperparameters
kernel_size = 20
num_filters = 8
batch_size = 12
pool = 36
dropout = 0.2
num_nodes = 250
num_epochs = 310
learning_rate = 0.001
beta = 0.95

# Get training data
curve_file = "../input/training_set.csv"

# Get dataframe containing metadata with zero redshift
meta_file = "../input/training_set_metadata.csv"
meta_df = pd.read_csv(meta_file)
ddf = meta_df['ddf'] == 0
gal = meta_df['hostgal_photoz'] != 0.0
meta_df = meta_df[ddf & gal]

# Get targets (6, 92, ...) and construct labels
targets = np.sort(meta_df['target'].unique(), axis=0)
num_classes = targets.size
label_range = range(0, num_classes)
target_dict = dict(zip(targets, label_range))
target_labels = [target_dict[x] for x in meta_df['target'].values]
label_array = np.array(target_labels)
labels = keras.utils.to_categorical(target_labels, num_classes=num_classes, dtype='int32')

# Global variables
accuracy = 0.0
conf_matrix = []
best_num_nodes = 0

def load_data():

    # Read curve data
    obj_ids = meta_df['object_id'].values
    curve_df = pd.read_csv(curve_file)
    curve_df = curve_df[curve_df['object_id'].isin(obj_ids)]
    curve_grps = curve_df.groupby('object_id')
    num_objects = len(curve_grps)

    # Allocate arrays
    curves = np.empty([num_objects, num_passbands, curve_width])
    freqs = np.empty([num_objects, num_passbands, freq_width])

    # Iterate through objects
    for obj_index, grp in enumerate(curve_grps):

        # Iterate through bands
        min_times = []
        interp_funcs = []
        for band in range(num_passbands):

            # Get times and flux values
            band_data = grp[1][grp[1]['passband'] == band]
            times = band_data['mjd'].values
            flux_vals = np.clip(band_data['flux'].values, a_min=0.001, a_max=None)

           # Get interpolation functions
            min_time = times.min()
            func = interpolate.interp1d(times, flux_vals, bounds_error=False, fill_value=(flux_vals[0], flux_vals[-1]))
            if band != 0:
                min_times.append(min_time)
            interp_funcs.append(func)

            # Get frequency data
            end_time = min_time + curve_time_duration
            interp_times = np.arange(min_time, end_time, curve_time_interval)
            fft = np.abs(np.fft.rfft(func(interp_times))[:-1])
            freqs[obj_index, band, :] = fft

            # Get frequency data
            end_time = min_time + curve_time_duration
            interp_times = np.arange(min_time, end_time, curve_time_interval)
            freqs[obj_index, band, :] = np.abs(np.fft.rfft(func(interp_times))[:-1])

        # Determine time interval
        start_time = sum(min_times)/len(min_times)
        end_time = start_time + curve_time_duration
        interp_times = np.arange(start_time, end_time, curve_time_interval)

        # Obtain curves
        for band in range(num_passbands):
            curves[obj_index, band, :] = interp_funcs[band](interp_times)

    curves = np.reshape(curves, (num_objects, num_passbands, curve_width, 1))
    freqs = np.reshape(freqs, (num_objects, num_passbands, freq_width, 1))
    return curves, freqs

def train(curves, freqs):

    np.random.seed(3)

    # Process curve data
    curve_data = Input(shape=(num_passbands, curve_width, 1))
    curve_conv = Conv2D(filters = num_filters, kernel_size = (num_passbands, kernel_size),
        input_shape=(num_passbands, curve_width, 1), activation='elu')(curve_data)
    curve_pool = AveragePooling2D(pool_size = (1, pool))(curve_conv)
    curve_flat = Flatten()(curve_pool)

    # Process freq data
    freq_data = Input(shape=(num_passbands, freq_width, 1))
    freq_conv = Conv2D(filters = num_filters, kernel_size = (num_passbands, kernel_size),
        input_shape=(num_passbands, freq_width, 1), activation='elu')(freq_data)
    freq_pool = AveragePooling2D(pool_size = (1, pool))(freq_conv)
    freq_flat = Flatten()(freq_pool)

    # Combine and analyze data
    flat = Concatenate()([curve_flat, freq_flat])
    hidden = Dense(num_nodes, activation='elu')(flat)
    drop = Dropout(dropout)(hidden)
    output = Dense(num_classes, activation='softmax',
        activity_regularizer=keras.regularizers.l2())(drop)
    model = keras.models.Model(inputs=[curve_data, freq_data], outputs=output)
    adam = keras.optimizers.Adam(lr=learning_rate, beta_1=beta, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer='adam',
        metrics=[keras.metrics.categorical_accuracy])

    # Launch the training process
    skf = StratifiedKFold(n_splits=num_splits)
    acc_array = np.empty(2)
    count = 0

    for train_index, test_index in skf.split(curves, label_array):

        np.random.seed(3)

        model.fit(x=[curves[train_index], freqs[train_index]], y=labels[train_index],
            batch_size=batch_size, epochs=num_epochs, verbose=0)
        tmp = model.predict(x=[curves[test_index], freqs[test_index]])
        results = np.asarray([np.argmax(line) for line in tmp])

        # Print parameters
        #if count == 0:
        #    print("Learning rate: {}".format(learning_rate))
        #    print("Beta: {}".format(beta))            

        #Check accuracy
        accur = sklearn.metrics.accuracy_score(label_array[test_index], results)
        if count == 2 and accur < 0.5:
            continue
        #tmp_matrix = confusion_matrix(label_array[test_index], results)
        print("Accuracy: {}".format(accur))
        count = count + 1

    # Save model to file
    model.save('wfd_ext.h5')

curves, freqs = load_data()

#learning_rate_list = [0.00075, 0.001, 0.00125]
#beta_list = [0.9, 0.95, 0.99]
#for learning_rate in learning_rate_list:
#    for beta in beta_list:
train(curves, freqs)

#batch_size_list = [6, 8, 10, 12, 14]
#for batch_size in batch_size_list:
#    train(curves, freqs, batch_size)

# Print results
#print("Best accuracy: {}".format(accuracy))
#print("Confusion matrix:\n {}".format(conf_matrix))