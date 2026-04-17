from instarNet_class import *
from network_functions import *
import numpy as np
import pickle

with open("instar.pkl", "rb") as f:
    instar = pickle.load(f)
    np.set_printoptions(precision=2, suppress=True)

    print(instar.weights)

    #Visualize(instar.weights)