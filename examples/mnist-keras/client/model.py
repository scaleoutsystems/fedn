import tensorflow as tf

from fedn.utils.helpers.helpers import get_helper

NUM_CLASSES = 10
HELPER_MODULE = "numpyhelper"
helper = get_helper(HELPER_MODULE)


def compile_model(img_rows=28, img_cols=28):
    """Compile the TF model.

    param: img_rows: The number of rows in the image
    type: img_rows: int
    param: img_cols: The number of rows in the image
    type: img_cols: int
    return: The compiled model
    type: keras.model.Sequential
    """
    # Set input shape
    input_shape = (img_rows, img_cols, 1)

    # Define model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=input_shape))
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"))
    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])
    return model


def save_parameters(model, out_path):
    """Save model parameters to file.

    :param model: The model to serialize.
    :type model: keras.model.Sequential
    :param out_path: The path to save the model to.
    :type out_path: str
    """
    weights = model.get_weights()
    helper.save(weights, out_path)


def load_parameters(model_path):
    """Load model parameters from file and populate model.

    :param model_path: The path to load from.
    :type model_path: str
    :return: The loaded model.
    :rtype: keras.model.Sequential
    """
    model = compile_model()
    weights = helper.load(model_path)
    model.set_weights(weights)
    return model


def init_seed(out_path="../seed.npz"):
    """Initialize seed model and save it to file.

    :param out_path: The path to save the seed model to.
    :type out_path: str
    """
    weights = compile_model().get_weights()
    helper.save(weights, out_path)


if __name__ == "__main__":
    init_seed("../seed.npz")
