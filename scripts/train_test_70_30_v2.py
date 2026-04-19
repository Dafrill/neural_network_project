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

print(f"Znaleziono {len(climate_folders)} folderow klimatycznych:")
for f in climate_folders:
    print(f"  - {f} -> {get_climate_group(f)}")

print("\n" + "="*60)
print("TWORZENIE SIECI INSTAR - 70% TRENINGOWE")
print("="*60)

train_ratio = 0.7

group_data_train = {'A': [], 'B': [], 'C': [], 'D': [], 'E': []}
group_data_test = {'A': [], 'B': [], 'C': [], 'D': [], 'E': []}

for folder in climate_folders:
    folder_path = os.path.join(klimatyczne_path, folder)
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    if len(csv_files) == 0:
        continue
    
    random.shuffle(csv_files)
    
    split_idx = int(len(csv_files) * train_ratio)
    train_files = csv_files[:split_idx]
    test_files = csv_files[split_idx:]
    
    group = get_climate_group(folder)
    
    for fname in train_files:
        try:
            df = pd.read_csv(os.path.join(folder_path, fname))
            vec = GetX_Vector(df)
            if not np.isnan(vec).any():
                group_data_train[group].append(vec)
        except:
            pass
    
    for fname in test_files:
        try:
            df = pd.read_csv(os.path.join(folder_path, fname))
            vec = GetX_Vector(df)
            if not np.isnan(vec).any():
                group_data_test[group].append(vec)
        except:
            pass
    
    print(f"{folder} ({group}): {len(train_files)} tren., {len(test_files)} test")

climate_dict = {}
for group in ['A', 'B', 'C', 'D', 'E']:
    if len(group_data_train[group]) > 0:
        climate_dict[group] = group_data_train[group][0]
    else:
        climate_dict[group] = [0.0] * 24

print("\n" + "="*60)
print("TRAINING SIECI INSTAR (5 grup: A,B,C,D,E)")
print("="*60)

instar = InstarNet(climate_dict, learning_rate=0.2)

for group, train_vectors in group_data_train.items():
    for vec in train_vectors:
        instar.train_winner(vec)

print("\n" + "="*60)
print("TESTOWANIE SIECI (30% danych)")
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
        if pred_label == group:
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

save_path = os.path.join(base_path, "instar_70_30_v2.pkl")
with open(save_path, "wb") as f:
    pickle.dump(instar, f)
print(f"\nZapisano siec do: {save_path}")