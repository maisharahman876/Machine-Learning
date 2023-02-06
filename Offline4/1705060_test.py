import pickle
import numpy as np

import pickle

# Load the model object from disk
with open("model.pkl", "rb") as file:
    model = pickle.load(file)
