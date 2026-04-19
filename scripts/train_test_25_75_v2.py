import numpy as np
import os
import pandas as pd
import pickle
import random
from network_functions import GetX_Vector
from instarNet_class import InstarNet

def get_climate_group(folder_name):
    folder_name = folder_name.upper()
    if folder_name.startswith('AF') or folder_name.startswith('AM') or folder_name.startswith('AW'):
        return 'A'
    elif folder_name.startswith('BW') or folder_name.startswith('BS') or folder_name.startswith('BH'):
        return 'B'
    elif folder_name.startswith('CFA') or folder_name.startswith('CFB') or folder_name.startswith('CFC') or \
         folder_name.startswith('CSA') or folder_name.startswith('CSB') or folder_name.startswith('CSC') or \
         folder_name.startswith('CWA') or folder_name.startswith('CWB') or folder_name.startswith('CWC'):
        return 'C'
    elif folder_name.startswith('DFA') or folder_name.startswith('DFB') or folder_name.startswith('DFC') or \
         folder_name.startswith('DSA') or folder_name.startswith('DSB') or folder_name.startswith('DSC') or \
         folder_name.startswith('DWA') or folder_name.startswith('DWB') or folder_name.startswith('DWC'):
        return 'D'
    elif folder_name.startswith('EF') or folder_name.startswith('ET'):
        return 'E'
    return folder_name[0] if folder_name else '?'

base_path = "/home/magda/Dokumenty/esi_projekt/neural_network_project"
klimatyczne_path = os.path.join(base_path, "klimatyczne")

climate_folders = [d for d in os.listdir(klimatyczne_path) if os.path.isdir(os.path.join(klimatyczne_path, d))]
climate_folders.sort()

print(f"Znaleziono {len(climate_folders)} folderow klimatycznych")

train_ratio = 0.25

all_group_vectors = {'A': [], 'B': [], 'C': [], 'D': [], 'E': []}

for folder in climate_folders:
    folder_path = os.path.join(klimatyczne_path, folder)
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    if len(csv_files) == 0:
        continue
    
    group = get_climate_group(folder)
    
    for fname in csv_files:
        try:
            df = pd.read_csv(os.path.join(folder_path, fname))
            vec = GetX_Vector(df)
            if not np.isnan(vec).any():
                all_group_vectors[group].append(vec)
        except:
            pass
    
    print(f"{folder} -> {group}: {len(csv_files)} plikow")

print("\n" + "="*60)
print("INICJALIZACJA WIECEJ NEURONOW (5 losowych wektorow per grupa)")
print("="*60)

climate_dict = {}

for group in ['A', 'B', 'C', 'D', 'E']:
    vectors = all_group_vectors[group]
    if len(vectors) >= 5:
        selected = random.sample(vectors, 5)
        for i, vec in enumerate(selected):
            climate_dict[f"{group}_{i}"] = vec
    elif len(vectors) > 0:
        climate_dict[group] = vectors[0]
    else:
        climate_dict[group] = [0.0] * 24
    print(f"Grupa {group}: {len([k for k in climate_dict.keys() if k.startswith(group)])} neuronow")

print("\n" + "="*60)
print("PODZIAL: 25% treningowe, 75% testowe")
print("="*60)

group_data_train = {'A': [], 'B': [], 'C': [], 'D': [], 'E': []}
group_data_test = {'A': [], 'B': [], 'C': [], 'D': [], 'E': []}

for group, vectors in all_group_vectors.items():
    n_train = int(len(vectors) * train_ratio)
    group_data_train[group] = vectors[:n_train]
    group_data_test[group] = vectors[n_train:]
    print(f"Grupa {group}: {len(group_data_train[group])} tren., {len(group_data_test[group])} test")

print("\n" + "="*60)
print("TRAINING SIECI INSTAR")
print("="*60)

instar = InstarNet(climate_dict, learning_rate=0.1)

for group, train_vectors in group_data_train.items():
    for vec in train_vectors:
        instar.train_winner(vec)

print("\n" + "="*60)
print("TESTOWANIE SIECI (75% danych)")
print("="*60)

correct = 0
total = 0
results_by_group = {'A': {'correct': 0, 'total': 0},
                  'B': {'correct': 0, 'total': 0},
                  'C': {'correct': 0, 'total': 0},
                  'D': {'correct': 0, 'total': 0},
                  'E': {'correct': 0, 'total': 0}}

for group, test_vectors in group_data_test.items():
    for vec in test_vectors:
        pred_idx, pred_label = instar.predict(vec)
        total += 1
        results_by_group[group]['total'] += 1
        if group in pred_label:
            correct += 1
            results_by_group[group]['correct'] += 1

print("\n" + "="*60)
print("WYNIKI TESTOWANIA WG GRUP KLIMATYCZNYCH:")
print("="*60)
for group in ['A', 'B', 'C', 'D', 'E']:
    res = results_by_group[group]
    acc = res['correct'] / res['total'] * 100 if res['total'] > 0 else 0
    print(f"Grupa {group}: {res['correct']}/{res['total']} ({acc:.1f}%)")

overall_accuracy = correct / total * 100 if total > 0 else 0
print(f"\nOGOLNA DOKLADNOSC: {overall_accuracy:.2f}%")

save_path = os.path.join(base_path, "instar_25_75_v2.pkl")
with open(save_path, "wb") as f:
    pickle.dump(instar, f)
print(f"\nZapisano siec do: {save_path}")