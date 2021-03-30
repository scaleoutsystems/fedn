# import keras as k
# from fedn.utils.kerasweights import KerasWeightsHelper
# from models.imdb_model import create_seed_model
#
#
# def predict(global_model, review):
#     print("-- RUNNING PREDICTION --", flush=True)
#     helper = KerasWeightsHelper()
#     weights = helper.load_model(global_model)
#
#     model = create_seed_model()
#     model.set_weights(weights)
#
#     # use model
#     d = k.datasets.imdb.get_word_index()
#     words = review.split()
#     review = []
#     for word in words:
#         if word not in d:
#             review.append(2)
#         else:
#             review.append(d[word] + 3)
#
#     review = k.preprocessing.sequence.pad_sequences([review], truncating='pre', padding='pre', maxlen=100)
#     prediction = model.predict(review)
#     return prediction[0][0]
#
#
# if __name__ == '__main__':
#
#     global_model = '7f06752e-66d4-4973-bd42-2c54490cb1c4' # global model name from minio repo
#     review = "Movie Review: Nothing was typical about this.\
#     Everything was beautifully done in this movie, the story, the flow,\
#     the scenario, everything. I highly recommend it for mystery lovers, \
#     for anyone who wants to watch a good movie!"
#     result = predict(global_model, review)
#     print("Prediction: (0 = negative, 1 = positive) = %0.4f" % result)
