import numpy as np
import os
import pandas as pd
import pickle
from network_functions import GetX_Vector
from instarNet_class import InstarNet

base_path = "/home/magda/Dokumenty/esi_projekt/neural_network_project"
klimatyczne_path = os.path.join(base_path, "klimatyczne")

climate_folders = [d for d in os.listdir(klimatyczne_path) if os.path.isdir(os.path.join(klimatyczne_path, d))]
climate_folders.sort()

print(f"Znaleziono {len(climate_folders)} folderow klimatycznych")

all_data = []
for folder in climate_folders:
    folder_path = os.path.join(klimatyczne_path, folder)
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    if len(csv_files) == 0:
        continue
    
    folder_upper = folder.upper()
    if folder_upper.startswith('AF') or folder_upper.startswith('AM') or folder_upper.startswith('AW'):
        climate_type = '1'  # Rownikowy
    elif folder_upper.startswith('BW') or folder_upper.startswith('BS'):
        climate_type = '3'  # Zwrotnikowy suchy
    elif folder_upper.startswith('CF'):
        climate_type = '6'  # Podzwrotnikowy morski
    elif folder_upper.startswith('CS') or folder_upper.startswith('CW'):
        climate_type = '6'  
    elif folder_upper.startswith('DF') or folder_upper.startswith('DS'):
        climate_type = '10'  # Umiarkowany kontynentalny
    elif folder_upper.startswith('DW'):
        climate_type = '8'  # Podzwrotnikowy Monsunowy
    elif folder_upper.startswith('EF') or folder_upper.startswith('ET'):
        climate_type = '14'  # Subpolarny/Tundra
    else:
        climate_type = '0'
    
    for fname in csv_files:
        try:
            df = pd.read_csv(os.path.join(folder_path, fname))
            vec = GetX_Vector(df)
            if not np.isnan(vec).any():
                all_data.append({'vector': vec, 'true_type': climate_type, 'folder': folder})
        except:
            pass

print(f"Lacznie probek: {len(all_data)}")

model_files = ['instar01.pkl', 'instar02.pkl', 'instarco4.pkl']

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
        
        results = {}
        correct = 0
        
        for item in all_data:
            vec = item['vector']
            true_type = item['true_type']
            
            if true_type == '0':
                continue
            
            pred_idx, pred_label = instar.predict(vec)
            
            if true_type not in results:
                results[true_type] = {'correct': 0, 'total': 0}
            results[true_type]['total'] += 1
            
            if true_type in pred_label:
                correct += 1
                results[true_type]['correct'] += 1
        
        print("\nWYNIKI:")
        types = ['1', '3', '6', '8', '10', '14']
        type_names = {'1': 'Rownikowy', '3': 'Zwrotnikowy', '6': 'Podzwrotnikowy', '8': 'Monsunowy', '10': 'Kontynentalny', '14': 'Tundra/Polarny'}
        
        for t in types:
            if t in results:
                res = results[t]
                acc = res['correct'] / res['total'] * 100 if res['total'] > 0 else 0
                print(f"  {type_names[t]}: {res['correct']}/{res['total']} ({acc:.1f}%)")
        
        print(f"\nOGOLNA DOKLADNOSC: {correct}/{len(all_data)} ({correct/len(all_data)*100:.2f}%)")
        
    except Exception as e:
        print(f"BŁĄD: {e}")