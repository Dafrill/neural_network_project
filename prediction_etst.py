# --- PLIK TESTOWY (np. test_v3.py) ---
import pickle
import pandas as pd
import os
from instarNet_class import InstarNet

from network_functions import GetX_Vector

# 1. Wczytaj przetrenowaną sieć
with open("instarco4.pkl", "rb") as f:
    instar = pickle.load(f)

# 2. Wybieranie pliku, którego nie było w treningu
folder_path = './cleaned_data'
all_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])
test_file = all_files[2]

# 3. Przetwórz dane i sprawdź wynik
df = pd.read_csv(os.path.join(folder_path, test_file))
station_data = GetX_Vector(df)

# Używamy predict, a nie train_winner!
index, climate_name = instar.predict(station_data)

print(f"PLIK: {test_file}")
print(f"ROZPOZNANY KLIMAT: {climate_name}")