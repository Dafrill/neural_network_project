import numpy as np
import os
import pandas as pd
import pickle
from network_functions import GetX_Vector
from instarNet_class import InstarNet

def get_climate_group(folder_name):
    folder_name = folder_name.upper()
    if folder_name.startswith('AF') or folder_name.startswith('AM') or folder_name.startswith('AW'):
        return 0
    elif folder_name.startswith('BW') or folder_name.startswith('BS'):
        return 1
    elif folder_name.startswith('CF') or folder_name.startswith('CS') or folder_name.startswith('CW'):
        return 2
    elif folder_name.startswith('DF') or folder_name.startswith('DS') or folder_name.startswith('DW'):
        return 3
    elif folder_name.startswith('EF') or folder_name.startswith('ET'):
        return 4
    return -1

base_path = "/home/magda/Dokumenty/esi_projekt/neural_network_project"
klimatyczne_path = os.path.join(base_path, "klimatyczne")

model_files = ['instar01.pkl', 'instar02.pkl', 'instarco4.pkl']

climate_folders = [d for d in os.listdir(klimatyczne_path) if os.path.isdir(os.path.join(klimatyczne_path, d))]

all_X = []
all_y = []

for folder in climate_folders:
    folder_path = os.path.join(klimatyczne_path, folder)
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    if len(csv_files) == 0:
        continue
    group = get_climate_group(folder)
    if group == -1:
        continue
    for fname in csv_files:
        try:
            df = pd.read_csv(os.path.join(folder_path, fname))
            vec = GetX_Vector(df)
            if not np.isnan(vec).any():
                all_X.append(vec)
                all_y.append(group)
        except:
            pass

all_X = np.array(all_X)
all_y = np.array(all_y)

print(f"Lacznie probek: {len(all_X)}")

for model_file in model_files:
    model_path = os.path.join(base_path, model_file)
    if not os.path.exists(model_path):
        print(f"\n!!! PLIK {model_file} NIE ISTNIEJE !!!")
        continue
    
    print(f"\n{'='*60}")
    print(f"MODEL: {model_file}")
    print("="*60)
    
    try:
        with open(model_path, 'rb') as f:
            instar = pickle.load(f)
        
        predictions = []
        for vec in all_X:
            pred_idx, pred_label = instar.predict(vec)
            predictions.append(pred_label)
        
        correct = 0
        results_by_group = {i: {'correct': 0, 'total': 0} for i in range(5)}
        
        for i, pred in enumerate(predictions):
            true_label = all_y[i]
            results_by_group[true_label]['total'] += 1
            
            pred_group = -1
            if 'A' in str(pred_label):
                pred_group = 0
            elif 'B' in str(pred_label):
                pred_group = 1
            elif 'C' in str(pred_label):
                pred_group = 2
            elif 'D' in str(pred_label):
                pred_group = 3
            elif 'E' in str(pred_label):
                pred_group = 4
            
            if pred_group == true_label:
                correct += 1
                results_by_group[true_label]['correct'] += 1
        
        group_names = ['A', 'B', 'C', 'D', 'E']
        print("\nWYNIKI WG GRUP:")
        for i in range(5):
            res = results_by_group[i]
            acc = res['correct'] / res['total'] * 100 if res['total'] > 0 else 0
            print(f"Grupa {group_names[i]}: {res['correct']}/{res['total']} ({acc:.1f}%)")
        
        overall_accuracy = correct / len(all_y) * 100
        print(f"\nOGOLNA DOKLADNOSC: {overall_accuracy:.2f}%")
        
    except Exception as e:
        print(f"BŁĄD: {e}")