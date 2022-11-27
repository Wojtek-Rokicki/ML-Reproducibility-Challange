from keras.datasets import mnist
from keras.utils import np_utils
from sklearn.utils import shuffle


def load_data():
    return mnist.load_data()


def preprocess_data(x, y, start=None, end=None, shuffle_data=False):
    x, y = shuffle(x, y, random_state=101)

    x = x.reshape(x.shape[0], 28 * 28, 1)
    x = x.astype("float64")
    x = x / 255  # reduce pixel values

    # oneHotEncode output which is a number in range [0,9] into a vector of size 10
    y = y.astype("float64")
    y = np_utils.to_categorical(y)
    y = y.reshape(y.shape[0], 10, 1)

    if start and end:
        return x[start:end], y[start:end]
    elif start:
        return x[start:], y[start:]
    elif end:
        return x[:end], y[:end]
    else:
        return x, y
