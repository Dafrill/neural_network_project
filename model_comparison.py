import pickle
import pandas as pd
import os
import numpy as np
from instarNet_class import InstarNet
from network_functions import GetX_Vector

models_to_check = {
    "Model (Każdy)": "instar.pkl",
    "Model (Co 3)": "instar02.pkl",
    "Model (Co 4)": "instarco4.pkl"
}

folder_path = './cleaned_data'
all_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])

# Wybieramy pliki, których modele nie widziały (co 10-ty plik)
test_files = all_files[::10]

results = []

# 2. Pętla porównawcza
for model_name, model_path in models_to_check.items():
    with open(model_path, "rb") as f:
        net = pickle.load(f)
    if not hasattr(net, 'labels'):
        from network_functions import all_climates
        net.labels = list(all_climates.keys())

    correct_hits = 0

    for filename in test_files:
        df = pd.read_csv(os.path.join(folder_path, filename))
        x = GetX_Vector(df)

        # Predykcja
        _, predicted_label = net.predict(x)

        if any(code in predicted_label for code in ["1N", "10N", "5S"]):
            correct_hits += 1

    accuracy = (correct_hits / len(test_files)) * 100
    results.append({"Model": model_name, "Accuracy": f"{accuracy:.2f}%"})

# 3. Wyświetlenie wyników
df_results = pd.DataFrame(results)
print(df_results)