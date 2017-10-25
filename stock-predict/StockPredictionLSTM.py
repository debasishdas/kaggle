from __future__ import print_function

from __future__ import print_function

import os
import sys
import numpy as np
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten, concatenate
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
import pandas as pd
from keras.layers.recurrent import LSTM
import pickle

def retainAlpha(word):
    for s in word:
        if ~s.isalpha():
            return False
    return True

BASE_DIR = '/Users/debasishdas/Documents/Personal/Coursera/NeuralNet/Projects/keras/data'
GLOVE_DIR = BASE_DIR + '/glove.6B/'
TEXT_DATA_DIR = BASE_DIR + '/20_newsgroup/'
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 5000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
NBR_OF_HDLNS = 5

# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

data = pd.read_csv('Data/stocknews/Combined_News_DJIA.csv')

train = data[data['Date'] < '2015-01-01']
test = data[data['Date'] > '2014-12-31']

train_labels = train["Label"].tolist()
test_labels = test["Label"].tolist()

tokenizers = {}
all_train_sequences = {}
all_test_sequences = {}

for index in range(1, 26):
    hdr = 'Top' + str(index)
    train_headlines = [' '.join(text_to_word_sequence(str(headline) if type(headline) != 'str' else headline)[1:])
                       for headline in train[hdr].tolist()]
    test_headlines = [' '.join(text_to_word_sequence(str(headline) if type(headline) != 'str' else headline)[1:])
                      for headline in test[hdr].tolist()]
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(train_headlines)
    tokenizers[index] = tokenizer
    all_train_sequences[index] = tokenizer.texts_to_sequences(train_headlines)
    all_test_sequences[index] = tokenizer.texts_to_sequences(test_headlines)

# One RNN for each headline
merged_lstm_output = []
merged_input = []
merged_padded_data_train = []
merged_padded_data_test = []

for hdln_index in range(1, NBR_OF_HDLNS):
    padded_data_train = pad_sequences(all_train_sequences[hdln_index], maxlen=MAX_SEQUENCE_LENGTH)
    padded_data_test = pad_sequences(all_test_sequences[hdln_index], maxlen=MAX_SEQUENCE_LENGTH)
    word_index = tokenizers[hdln_index].word_index
    print('Found %s unique tokens.' % len(word_index))
    print('Max index value in word_index = %s' % max(word_index.values()))
    print('Min index value in word_index = %s' % min(word_index.values()))
    num_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    print("Embedding Matrix dimensions = "+ str(embedding_matrix.shape[0]) + "," + str(embedding_matrix.shape[1]))
    for word, i in word_index.items():
        if i >= MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    lstm_layer = LSTM(64)
    lstm_output = lstm_layer(embedded_sequences)
    merged_padded_data_train.append(padded_data_train)
    merged_padded_data_test.append(padded_data_test)
    merged_lstm_output.append(lstm_output)
    merged_input.append(sequence_input)

merged_vector = concatenate(merged_lstm_output, axis=-1)
predictions = Dense(1, activation='sigmoid')(merged_vector)

model = Model(inputs=merged_input, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(merged_padded_data_train, train_labels, epochs=10)

#Saving the Model to disk.
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# Evaluate Model performance
model.evaluate(x=merged_padded_data_test, y=test_labels)

# Testing the model
pred_labels = model.predict(merged_padded_data_test, verbose=1)
pred_series = pd.Series(pred_labels)
test["pred_labels"] = pred_series.values
test.to_csv("test-results.csv")

# Loading model from disk
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")


