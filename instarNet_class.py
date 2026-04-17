import numpy as np
from network_functions import *
import os
import pandas as pd



class InstarNet:
    def __init__(self, climate_dict, learning_rate=0.1):
        
        
        self.weights = np.vstack([np.array(v, dtype=float) for v in climate_dict.values()])

        self.eta = learning_rate

    def predict(self, x):
        """Look for most suitable weights."""
        
        activations = np.dot(self.weights, x)
        winner_index = np.argmax(activations)
        return winner_index

    def train_winner(self, x):
        """Find the best suitable weights and update them"""
        index = self.predict(x)
        self.weights[index] += self.eta * (np.array(x) - self.weights[index])
        print("weights updated")
        return f"Updated"



if __name__ == "__main__":
    instar = InstarNet(all_climates, learning_rate=0.2)
  
    np.set_printoptions(precision=2, suppress=True)

    folder_path = './cleaned_data'  

    for filename in os.listdir(folder_path):
      
            
            
        
        df = pd.read_csv(f"{folder_path}/{filename}")
            
            
            
        #     #
        try:
            station_data = GetX_Vector(df)
            if np.isnan(station_data).any():
                
                continue

            instar.train_winner(station_data)
            #print("network updated")
                
        except Exception as e:
            print(f"Błąd w pliku {filename}: {e}")
    
    import pickle


    with open("instar.pkl", "wb") as f:
        pickle.dump(instar, f)

