import numpy as np
from network_functions import *
import os
import pandas as pd



class InstarNet:
    def __init__(self, climate_dict, learning_rate=0.1):
        self.labels = list(climate_dict.keys())
        # Pobieramy wartości (wektory wag)
        self.weights = np.vstack([np.array(v, dtype=float) for v in climate_dict.values()])
        self.eta = learning_rate

    def predict(self, x):
        """Look for most suitable weights - najmniejsza odleglosc."""
        x = np.array(x)
        distances = np.linalg.norm(self.weights - x, axis=1)
        winner_index = np.argmin(distances)
        return winner_index, self.labels[winner_index]

    def train_winner(self, x):
        """Find the best suitable weights and update them"""
        index, label = self.predict(x)
        self.weights[index] += self.eta * (np.array(x) - self.weights[index])
        print(f"Zaktualizowano wzorzec dla klimatu: {label}")
        return label



if __name__ == "__main__":
    import zipfile


    instar = InstarNet(all_climates, learning_rate=0.2)
  
    np.set_printoptions(precision=2, suppress=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    zip_path = os.path.join(script_dir, 'cleaned_data.zip')
    folder_path = os.path.join(script_dir, 'cleaned_data')

    if not os.path.exists(folder_path):
        if os.path.exists(zip_path):
            print("Rozpakowuję dane do folderu...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(script_dir)
            print("Gotowe! Dane wypakowane.")
        else:
            print("Błąd: Nie znalazłem pliku ZIP ani folderu!")
            exit()  # Kończymy, jeśli nie ma skąd wziąć danych

    all_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])

    for filename in all_files[::4]:
            
        #     #
        try:
            df = pd.read_csv(os.path.join(folder_path, filename))

            station_data = GetX_Vector(df)
            if np.isnan(station_data).any():
                continue

            recognized_climate = instar.train_winner(station_data)
        #print("network updated")
                
        except Exception as e:
            print(f"Błąd w pliku {filename}: {e}")
    
    import pickle


    with open("instarco4.pkl", "wb") as f:
        pickle.dump(instar, f)

