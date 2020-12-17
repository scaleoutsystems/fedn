from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding, LSTM
from tensorflow.keras.layers import Conv1D, MaxPooling1D


def create_seed_model():
    """
    Helper function to generate an initial seed model.
    Define CNN-LSTM architecture
    :return: model
    """
    # Embedding
    max_features = 20000
    maxlen = 100
    embedding_size = 128

    # Convolution
    kernel_size = 5
    filters = 64
    pool_size = 4

    # LSTM
    lstm_output_size = 70

    model = Sequential()
    model.add(Embedding(max_features, embedding_size, input_length=maxlen))
    model.add(Dropout(0.25))
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(LSTM(lstm_output_size))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


if __name__ == '__main__':
    # Create a seed model and push to Minio
    model = create_seed_model()
    outfile_name = "879fa112-c861-4cb1-a25d-775153e5b550"
    model.save(outfile_name, save_format='h5')