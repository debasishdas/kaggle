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
from keras.models import model_from_json
from sklearn.model_selection import train_test_split

def retainAlpha(word):
    for s in word:
        if ~s.isalpha():
            return False
    return True

# Download the Stanford Glove word vectors,
# unzip and save it under <root>/data/glove.6B/
BASE_DIR = '../data'
GLOVE_DIR = BASE_DIR + '/glove.6B/'
MAX_SEQUENCE_LENGTH = 60
MAX_NB_WORDS = 50000
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

train, valid = train_test_split(pd.read_csv('Data/train.csv'), test_size = VALIDATION_SPLIT)

test = pd.read_csv('Data/test.csv')

train_labels = pd.to_numeric(train["is_duplicate"]).tolist()
valid_labels = pd.to_numeric(valid["is_duplicate"]).tolist()

train_question1 = [' '.join(text_to_word_sequence(str(question) if type(question) != 'str' else question))
                   for question in train["question1"].tolist()]
train_question2 = [' '.join(text_to_word_sequence(str(question) if type(question) != 'str' else question))
                   for question in train["question2"].tolist()]

valid_question1 = [' '.join(text_to_word_sequence(str(question) if type(question) != 'str' else question))
                   for question in valid["question1"].tolist()]
valid_question2 = [' '.join(text_to_word_sequence(str(question) if type(question) != 'str' else question))
                   for question in valid["question2"].tolist()]

test_question1 = [' '.join(text_to_word_sequence(str(question) if type(question) != 'str' else question))
                   for question in test["question1"].tolist()]
test_question2 = [' '.join(text_to_word_sequence(str(question) if type(question) != 'str' else question))
                   for question in test["question2"].tolist()]


tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(train_question1 + train_question2 + valid_question1 + valid_question2)
train_q1_seq = tokenizer.texts_to_sequences(train_question1)
train_q2_seq = tokenizer.texts_to_sequences(train_question2)
valid_q1_seq = tokenizer.texts_to_sequences(valid_question1)
valid_q2_seq = tokenizer.texts_to_sequences(valid_question2)
test_q1_seq = tokenizer.texts_to_sequences(test_question1)
test_q2_seq = tokenizer.texts_to_sequences(test_question2)

# One RNN for each headline
merged_lstm_output = []
merged_input = []
merged_padded_data_train = []
merged_padded_data_test = []

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
print('Max index value in word_index = %s' % max(word_index.values()))
print('Min index value in word_index = %s' % min(word_index.values()))
num_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
print("Embedding Matrix dimensions = " + str(embedding_matrix.shape[0]) + "," + str(embedding_matrix.shape[1]))
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


padded_q1_train = pad_sequences(train_q1_seq, maxlen=MAX_SEQUENCE_LENGTH)
padded_q2_train = pad_sequences(train_q2_seq, maxlen=MAX_SEQUENCE_LENGTH)
padded_q1_valid = pad_sequences(valid_q1_seq, maxlen=MAX_SEQUENCE_LENGTH)
padded_q2_valid = pad_sequences(valid_q2_seq, maxlen=MAX_SEQUENCE_LENGTH)
padded_q1_test = pad_sequences(test_q1_seq, maxlen=MAX_SEQUENCE_LENGTH)
padded_q2_test = pad_sequences(test_q2_seq, maxlen=MAX_SEQUENCE_LENGTH)

# Combining the padded dataset for being used as inputs to separate input layers.
padded_data_train = [padded_q1_train, padded_q2_train]
padded_data_valid = [padded_q1_valid, padded_q2_valid]
padded_data_test = [padded_q1_test, padded_q2_test]

# Separate input layers for question1 and question2
sequence_input1 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
sequence_input2 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

# Combining the input layers for model definition
seq_input = [sequence_input1, sequence_input2]

# Embedding layers
embedded_sequences1 = embedding_layer(sequence_input1)
embedded_sequences2 = embedding_layer(sequence_input2)

# Shared LSTM layer for each question
shared_lstm_layer = LSTM(64)
lstm_output1 = shared_lstm_layer(embedded_sequences1)
lstm_output2 = shared_lstm_layer(embedded_sequences2)

# Merging the LSTM outputs and concatenating them
# to be fed into the final logistic layer
merged_lstm_output = [lstm_output1, lstm_output2]
merged_vector = concatenate(merged_lstm_output, axis=-1)

# The final logistic output layer
predictions = Dense(1, activation='sigmoid')(merged_vector)

# Model definition
model = Model(inputs=seq_input, outputs=predictions)

# Compiling and training model
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(merged_padded_data_train, train_labels, epochs=50)

# Evaluate Model performance on validation data
loss, acc = model.evaluate(x=padded_data_valid, y=valid_labels)
print("Loss on validation data = %s and accuracy = %s" % (loss, acc))

# Saving the predictions on validation data
pred_labels_valid = model.predict(padded_data_valid, verbose=1)
pred_series_valid = pd.Series(pred_labels_valid.flat)
valid["pred_labels"] = pred_series_valid.values
valid.to_csv("quora-validation-results.csv")

# Save the model to disk
model_json = model.to_json()
with open("quora-shared.lstm.model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("quora-shared.lstm.model.h5")
print("Saved model to disk")

# Saving the predictions on test data
pred_labels_test = model.predict(padded_data_test, verbose=1)
pred_series_test = pd.Series(pred_labels_test.flat)
test["is_duplicate"] = pred_series_test.values
test.to_csv("quora-test-results.csv")


