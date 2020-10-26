import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GlobalMaxPooling1D, Embedding
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D

from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np


def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]
    return embedding_matrix


# Create an initial CNN Model
def create_seed_model():
    EMBEDDING_DIM = 50
    MAX_SEQUENCE_LENGTH = 100

    tokenizer = Tokenizer(num_words=100000)
    embedding_matrix = create_embedding_matrix('../data/word_embeddings/glove.6B.50d.txt',
                                               tokenizer.word_index,
                                               EMBEDDING_DIM)
    model = Sequential()
    model.add(Embedding(len(tokenizer.word_index) + 1, EMBEDDING_DIM,
                        weights=[embedding_matrix],
                        input_length=MAX_SEQUENCE_LENGTH,
                        trainable=False))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model


if __name__ == '__main__':
    # import keras
    # import tensorflow as tf
    # print('TF', tf.__version__)
    # print('KERAS', keras.__version__)
    # Create a seed model and push to Minio
    model = create_seed_model()
    outfile_name = "879fa112-c861-4cb1-a25d-775153e5b550"
    model.save(outfile_name, save_format='h5')
