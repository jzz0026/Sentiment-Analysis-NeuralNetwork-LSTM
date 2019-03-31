

import numpy
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

numpy.random.seed(7)


# Funtion to load cleaned reviews and perform train and test data plit load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)


# truncate and pad input sequences. we need to truncate and pad the input sequences so that they are all the same length for modeling. The model will learn the zero values carry no information so indeed the sequences are not the same length in terms of content, but same length vectors is required to perform the computation in Keras.
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

""" The first layer is the Embedded layer that uses 32 length vectors to represent each word. The next 
layer is the LSTM layer with 100 memory units (smart neurons). Finally, because this is a classification 
problem we use a Dense output layer with a single neuron and a sigmoid activation function to make 0 or 1 
predictions for the two classes (pos and neg) in the problem."""


