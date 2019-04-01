import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence



numpy.random.seed(9)
word_embed=os.path.abspath(os.path.join(root_path, 'data/word2embed.txt'))

# Funtion to load cleaned and vectorized reviews
#  but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = word_embed([0:top_words])
max_review_length = 500
embedding_vecor_length = 32

#LSTMs have the tendency to overfit and we need to apply regularization
#some hyperparameter tuning is needed
#Inside LSTM function we tune dropout setting
#Dropuout: randomly selects nodes to be dropped-out with a given probability (e.g. 20%) each weight update cycle. 
#Dropout is only used during the training of a model and is not used when evaluating the skill of the model.

numpy.random.seed(9)
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = word_embed([0:top_words])
# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=3, batch_size=64)
# Evaluation of the model calculating accuracy
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
