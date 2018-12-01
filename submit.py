from more_itertools import peekable
import numpy as np
from scipy import interpolate
import keras
import sys

# Constants
num_passbands = 6
curve_width = 1024
freq_width = curve_width//2
curve_time_duration = 857.6
curve_time_interval = curve_time_duration/float(curve_width)

# Create results file and write first line
result_file  = open('submission.csv', 'w')
result_file.write('object_id,class_6,class_15,class_16,class_42,class_52,class_53,class_62,class_64,class_65,class_67,class_88,class_90,class_92,class_95,class_99\n')

# Access metadata
curve_file = peekable(open('../input/test_set.csv', 'r'))
meta_file = open('../input/test_set_metadata.csv', 'r')

# Access models
ddf_gal_model = keras.models.load_model('ddf_gal.h5')
wfd_gal_model = keras.models.load_model('wfd_gal.h5')
ddf_ext_model = keras.models.load_model('ddf_ext.h5')
wfd_ext_model = keras.models.load_model('wfd_ext.h5')

# Iterate through lines in metadata file
next(meta_file)
next(curve_file)

meta_data = {}
for meta_line in meta_file:

    # Get metadata fields
    meta_fields = [x.strip() for x in meta_line.split(',')]

    # Check redshift
    if float(meta_fields[-4]) == 0.0:
        gal = True
    else:
        gal = False

    # Check DDF
    if int(meta_fields[5]) == 1:
        ddf = True
    else:
        ddf = False
        
    obj = int(meta_fields[0])
    meta_data[obj] = [gal, ddf]
    
meta_file.close()    

# Read curve file
obj_id = -1
flux_vals = {}
times = {}
for curve_line in curve_file:

    # Read fields
    curve_fields = [x.strip() for x in curve_line.split(',')]

    # Check for new object
    if obj_id == -1:
        obj_id = int(curve_fields[0])
        gal = meta_data[obj_id][0]
        ddf = meta_data[obj_id][1]
        result_file.write('{},'.format(obj_id))
        for band in range(num_passbands):
            flux_vals[band] = []
            times[band] = []

    # Read time/flux data
    times[int(curve_fields[2])].append(float(curve_fields[1]))
    flux_vals[int(curve_fields[2])].append(np.clip(float(curve_fields[3]), a_min=0.001, a_max=None))

    # Get curve fields
    try:
        next_line = curve_file.peek()
        next_fields = [x.strip() for x in next_line.split(',')]
        next_obj = int(next_fields[0])
    except StopIteration:
        next_obj = -2

    if next_obj != obj_id:

        # Iterate through bands
        min_times = []
        interp_funcs = []
        output_vals = np.zeros((15,), dtype=int)
        nan = False

        for band in range(num_passbands):

            # Create NumPy arrays
            time_array = np.array(times[band])
            flux_array = np.array(flux_vals[band])

            if time_array.size < 2 or flux_array.size < 2:
                nan = True
                break

            curves = np.empty([num_passbands, curve_width])
            freqs = np.empty([num_passbands, freq_width])

            # Get interpolation functions
            min_time = time_array.min()
            func = interpolate.interp1d(time_array, flux_array, bounds_error=False, fill_value=(flux_array[0], flux_array[-1]))
            if band != 0:
                min_times.append(min_time)
            interp_funcs.append(func)

            # Get frequency data
            end_time = min_time + curve_time_duration
            interp_times = np.arange(min_time, end_time, curve_time_interval)
            freqs[band, :] = np.abs(np.fft.rfft(func(interp_times))[:-1])

        if not nan:

            # Determine time interval
            start_time = sum(min_times)/len(min_times)
            end_time = start_time + curve_time_duration
            interp_times = np.arange(start_time, end_time, curve_time_interval)

            # Obtain curves
            for band in range(num_passbands):
                curves[band, :] = interp_funcs[band](interp_times)

            # Reshape curves and frequencies
            curves = np.reshape(curves, (1, num_passbands, curve_width, 1))
            freqs = np.reshape(freqs, (1, num_passbands, freq_width, 1))

            # Obtain prediction
            if ddf:
                if gal:
                    prediction = ddf_gal_model.predict(x=[curves, freqs])[0]
                else:
                    prediction = ddf_ext_model.predict(x=[curves, freqs])[0]

            else:
                if gal:
                    prediction = wfd_gal_model.predict(x=[curves, freqs])[0]
                else:
                    prediction = wfd_ext_model.predict(x=[curves, freqs])[0]

            index = np.argmax(prediction)
            nan = np.any(np.isnan(prediction))

        if nan:
            output_vals[11] = 1
        elif prediction[index] < 0.5:
            output_vals[14] = 1
        else:
            if gal:
                if index == 0:
                    output_vals[0] = 1
                elif index == 1:
                    output_vals[2] = 1
                elif index == 2:
                    output_vals[5] = 1
                elif index == 3:
                    output_vals[8] = 1
                elif index == 4:
                    output_vals[12] = 1
                else:
                    output_vals[14] = 1

            else:
                if index == 0:
                    output_vals[1] = 1
                elif index == 1:
                    output_vals[3] = 1
                elif index == 2:
                    output_vals[4] = 1
                elif index == 3:
                    output_vals[6] = 1
                elif index == 4:
                    output_vals[7] = 1
                elif index == 5:
                    output_vals[9] = 1
                elif index == 6:
                    output_vals[10] = 1
                elif index == 7:
                    output_vals[11] = 1
                elif index == 8:
                    output_vals[13] = 1
                else:
                    output_vals[14] = 1

        output_str = np.array2string(output_vals, separator=',')[1:-1]
        result_file.write('{}\n'.format(output_str))
        obj_id = -1