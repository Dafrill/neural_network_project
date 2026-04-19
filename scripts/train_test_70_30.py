import numpy as np
import os
import pandas as pd
import pickle
import random
from network_functions import GetX_Vector
from instarNet_class import InstarNet

base_path = "/home/magda/Dokumenty/esi_projekt/neural_network_project"
klimatyczne_path = os.path.join(base_path, "klimatyczne")

climate_folders = [d for d in os.listdir(klimatyczne_path) if os.path.isdir(os.path.join(klimatyczne_path, d))]
climate_folders.sort()

print(f"Znaleziono {len(climate_folders)} folderow klimatycznych:")
for f in climate_folders:
    print(f"  - {f}")

print("\n" + "="*60)
print("TWORZENIE SIECI INSTAR - 70% TRENINGOWE")
print("="*60)

climate_dict = {}
train_data_by_climate = {}
test_data_by_climate = {}

train_ratio = 0.7

for folder in climate_folders:
    folder_path = os.path.join(klimatyczne_path, folder)
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    if len(csv_files) == 0:
        print(f"  [OSTRZEZENIE] Folder {folder} jest pusty - pomijam")
        continue
    
    random.shuffle(csv_files)
    
    split_idx = int(len(csv_files) * train_ratio)
    train_files = csv_files[:split_idx]
    test_files = csv_files[split_idx:]
    
    train_vectors = []
    test_vectors = []
    
    for fname in train_files:
        try:
            df = pd.read_csv(os.path.join(folder_path, fname))
            vec = GetX_Vector(df)
            if not np.isnan(vec).any():
                train_vectors.append(vec)
        except:
            pass
    
    for fname in test_files:
        try:
            df = pd.read_csv(os.path.join(folder_path, fname))
            vec = GetX_Vector(df)
            if not np.isnan(vec).any():
                test_vectors.append(vec)
        except:
            pass
    
    climate_label = folder
    
    if len(train_vectors) > 0:
        first_vec = train_vectors[0]
    else:
        first_vec = [0.0] * 24
    
    climate_dict[climate_label] = first_vec
    train_data_by_climate[climate_label] = train_vectors
    test_data_by_climate[climate_label] = test_vectors
    
    print(f"{climate_label}: {len(train_vectors)} treningowych, {len(test_vectors)} testowych")

print("\n" + "="*60)
print("TRAINING SIECI INSTAR")
print("="*60)

instar = InstarNet(climate_dict, learning_rate=0.2)

for climate_label, train_vectors in train_data_by_climate.items():
    for vec in train_vectors:
        instar.train_winner(vec)

print("\n" + "="*60)
print("TESTOWANIE SIECI (30% danych)")
print("="*60)

save_path = os.path.join(base_path, "instar_70_30.pkl")
with open(save_path, "wb") as f:
    pickle.dump(instar, f)
print(f"Zapisano siec do: {save_path}")

correct = 0
total = 0
results_by_climate = {}

for climate_label, test_vectors in test_data_by_climate.items():
    if not test_vectors:
        continue
    
    correct_climate = 0
    for vec in test_vectors:
        pred_idx, pred_label = instar.predict(vec)
        total += 1
        if climate_label in pred_label:
            correct += 1
            correct_climate += 1
    
    results_by_climate[climate_label] = {
        'correct': correct_climate,
        'total': len(test_vectors)
    }

print("\n" + "="*60)
print("WYNIKI TESTOWANIA:")
print("="*60)
for climate_label, res in results_by_climate.items():
    acc = res['correct'] / res['total'] * 100 if res['total'] > 0 else 0
    print(f"{climate_label}: {res['correct']}/{res['total']} ({acc:.1f}%)")

overall_accuracy = correct / total * 100 if total > 0 else 0
print(f"\nOGOLNA DOKLADNOSC: {overall_accuracy:.2f}%")