# Kaggle's 2018 PLAsTiCC Astronomical Classification Challenge

This repo contains the code that I used to compete in Kaggle's [2018 PLAsTiCC Astronomical Classification challenge](https://www.kaggle.com/c/PLAsTiCC-2018). The goal is to read a series of records and classify celestial bodies into one of fifteen categories. Using Python and Keras, I created four deep neural networks (DNNs) to perform classification:
* ddf_gal_dnn - Classifies bodies measured using DDF (Deep Drilling Fields) in the Milky Way galaxy
* ddf_ext_dnn - Classifies bodies measured using DDF (Deep Drilling Fields) outside the Milky Way galaxy
* wfd_gal_dnn - Classifies bodies measured using WFD (Wide-Fast-Deep) in the Milky Way galaxy
* wfd_ext_dnn - Classifies bodies measured using WFD (Wide-Fast-Deep) outside the Milky Way galaxy

Each neural network analyzes two portions of data. The first contains the light curves for each of the six frequency bands (u, g, r, i, z, y) measured for the body. The second portion contains the frequency content of the light curves in the first portion. I computed the frequency content using NumPy's real Fast Fourier Transform (rfft) function.

I trained and tested the DNNs using 10-fold stratified cross-validation. This splits training data into ten groups and uses nine groups for training, one for testing. The process performs ten iterations to ensure that every group has been used for training and testing.

The submit.py script reads the four Keras models and the test files provided by Kaggle. Then it obtains a prediction for each record and writes the result to a CSV file called submission.csv.
