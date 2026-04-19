from instarNet_class import *
from network_functions import *
import numpy as np
import pickle

with open("instarco4.pkl", "rb") as f:
    instar = pickle.load(f)
    np.set_printoptions(precision=2, suppress=True)

    print(instar.weights)
    for i, label in enumerate(instar.labels):
        print(f"Klimat: {label}")
        print(f"Wagi (Temp + Opad): {instar.weights[i]}")
        print("-" * 30)
    #Visualize(instar.weights)