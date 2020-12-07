import pandas as pd
from sklearn.model_selection import train_test_split
# for text
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import yaml


def read_data(filename):
    """ Helper function to read and preprocess data for training with Keras. """

    with open('settings.yaml', 'r') as fh:
        try:
            settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError as e:
            raise e

    # The entire dataset is 50k comment, we can subsample here for quicker testing.
    df = pd.read_csv(filename, sep=',')
    X = df['text'].values
    y = df['label'].values
    x_train, x_test, y_train, y_test = train_test_split(
        X, y,
        test_size=settings['test_size'],
        random_state=settings['random_state'])
    # tokenizer create tokens for every word in the data corpus and map them to a index using dictionary.
    # word_index contains the index for each word
    tokenizer = Tokenizer(num_words=settings['num_words'])
    tokenizer.fit_on_texts(x_train)
    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    # Since we are going to build a sequence model. We should feed in a sequence of numbers to it.
    # And also we should ensure there is no variance in input shapes of sequences. It all should be of same lenght.
    # But texts in tweets have different count of words in it. To avoid this, we seek a little help from pad_sequence
    # to do our job. It will make all the sequence in one constant length max_sequence_lenght
    # Only consider the first 100 words of each movie review
    x_train = pad_sequences(x_train, padding='post', maxlen=settings['max_sequence_lenght'])
    x_test = pad_sequences(x_test, padding='post', maxlen=settings['max_sequence_lenght'])
    return x_train, x_test, y_train, y_test
