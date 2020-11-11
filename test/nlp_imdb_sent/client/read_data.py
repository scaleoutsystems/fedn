import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer


TAG_RE = re.compile(r'<[^>]+>')


def remove_tags(text):
    return TAG_RE.sub('', text)


def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence


# # removing the stopwords
# def remove_stopwords(text, is_lower_case=False):
#     # set stopwords to english
#     stopword_list = set(stopwords.words('english'))
#
#     # Tokenization of text
#     tokenizer = ToktokTokenizer()
#     tokens = tokenizer.tokenize(text)
#     tokens = [token.strip() for token in tokens]
#     if is_lower_case:
#         filtered_tokens = [token for token in tokens if token not in stopword_list]
#     else:
#         filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
#     filtered_text = ' '.join(filtered_tokens)
#     return filtered_text


def read_data(filename):
    """ Helper function to read and preprocess data for training with Keras. """
    # The entire dataset is 50k comment, we can subsample here for quicker testing.
    df = pd.read_csv(filename, sep=',')

    X = []
    sentences = list(df['text'])
    for sen in sentences:
        X.append(preprocess_text(sen))

    y = df['label']
    y = np.array(list(map(lambda x: 1 if x == "positive" else 0, y)))

    x_train, x_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42)

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

    max_sequence_lenght = 100 # Only consider the first 100 words of each movie review
    x_train = pad_sequences(x_train, padding='post', maxlen=max_sequence_lenght)
    x_test = pad_sequences(x_test, padding='post', maxlen=max_sequence_lenght)
    return x_train, x_test, y_train, y_test, tokenizer