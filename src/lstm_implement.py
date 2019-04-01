import numpy
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

root_path = Path(__file__).parents[2]
word_embed=os.path.abspath(os.path.join(root_path, 'data/word2embed.txt'))

numpy.random.seed(9)
# Funtion to load cleaned and vectorizzed reviews
#  but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = word_embed(num_words=top_words)

# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = word_embed(num_words=top_words)

#truncate and pad input sequences
#We need to truncate and pad the input sequences so all are same length
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

""" The first layer is the Embedded layer that uses 32 length vectors to represent each word. The next 
layer is the LSTM layer with 100 memory units (smart neurons). Finally, because this is a classification 
problem we use a Dense output layer with a single neuron and a sigmoid activation function to make 0 or 1 
predictions for the two classes (pos and neg) in the problem."""
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)


# Evaluate model on unseen data by calculating accuracy score
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))



