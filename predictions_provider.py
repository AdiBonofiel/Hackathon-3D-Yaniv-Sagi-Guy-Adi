import utils
import tensorflow as tf
import numpy as np


def provide_predictions(path_to_PDB):
    NUM_OF_MODELS = 10
    PATH_TO_MODELS = './models'
    predictions = []
    protein_to_predict = utils.generate_input(path_to_PDB)
    for i in range(NUM_OF_MODELS):
        model = tf.keras.models.load_model("%s/model%d.tf" %(PATH_TO_MODELS, i))
        prediction = model.predict(protein_to_predict[np.newaxis, :, :])
        predictions.append(prediction)
    return predictions
