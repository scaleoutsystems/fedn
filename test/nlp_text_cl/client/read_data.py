import pandas as pd
from sklearn.model_selection import train_test_split
# for text
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def read_data(filename):
    """ Helper function to read and preprocess data for training with Keras. """

    # The entire dataset is 50k comment, we can subsample here for quicker testing.
    df = pd.read_csv(filename, sep=',')
    sentences = df['text'].values
    y = df['label'].values
    x_train, x_test, y_train, y_test = train_test_split(
        sentences, y,
        test_size=0.25,
        random_state=1000)
    # tokenizer create tokens for every word in the data corpus and map them to a index using dictionary.
    # word_index contains the index for each word
    tokenizer = Tokenizer(num_words=100000)
    tokenizer.fit_on_texts(x_train)
    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    # Since we are going to build a sequence model. We should feed in a sequence of numbers to it.
    # And also we should ensure there is no variance in input shapes of sequences. It all should be of same lenght.
    # But texts in tweets have different count of words in it. To avoid this, we seek a little help from pad_sequence
    # to do our job. It will make all the sequence in one constant length MAX_SEQUENCE_LENGTH

    MAX_SEQUENCE_LENGTH = 100
    x_train = pad_sequences(x_train, padding='post', maxlen=MAX_SEQUENCE_LENGTH)
    x_test = pad_sequences(x_test, padding='post', maxlen=MAX_SEQUENCE_LENGTH)
    return x_train, x_test, y_train, y_test


