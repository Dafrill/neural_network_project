import numpy as np
import os
import pandas as pd
import pickle
from network_functions import GetX_Vector

class ImprovedNN:
    def __init__(self, input_size, hidden_size, output_size, class_weights=None, learning_rate=0.05):
        self.lr = learning_rate
        self.class_weights = class_weights
        np.random.seed(42)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, output):
        m = X.shape[0]
        delta2 = (output - y) * self.sigmoid_derivative(output)
        
        if self.class_weights is not None:
            for i in range(m):
                cls = np.argmax(y[i])
                delta2[i] *= self.class_weights[cls]
        
        dW2 = np.dot(self.a1.T, delta2) / m
        db2 = np.sum(delta2, axis=0, keepdims=True) / m
        
        delta1 = np.dot(delta2, self.W2.T) * self.relu_derivative(self.a1)
        dW1 = np.dot(X.T, delta1) / m
        db1 = np.sum(delta1, axis=0, keepdims=True) / m
        
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
    
    def train(self, X, y, epochs=100):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            if (epoch + 1) % 25 == 0:
                loss = np.mean((output - y) ** 2)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
    
    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)


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

climate_folders = [d for d in os.listdir(klimatyczne_path) if os.path.isdir(os.path.join(klimatyczne_path, d))]
climate_folders.sort()

print(f"Znaleziono {len(climate_folders)} folderow klimatycznych")

train_ratio = 0.7

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
    print(f"{folder} -> {['A','B','C','D','E'][group]}: {len(csv_files)} plikow")

all_X = np.array(all_X)
all_y = np.array(all_y)

print(f"\nRozklad klas: {np.bincount(all_y)}")

class_counts = np.bincount(all_y)
class_weights = {i: len(all_X) / (5 * class_counts[i]) if class_counts[i] > 0 else 1.0 for i in range(5)}
print(f"Wagi klas: {[f'{w:.2f}' for w in class_weights.values()]}")

indices = np.arange(len(all_X))
np.random.shuffle(indices)
all_X = all_X[indices]
all_y = all_y[indices]

split_idx = int(len(all_X) * train_ratio)
X_train, X_test = all_X[:split_idx], all_X[split_idx:]
y_train, y_test = all_y[:split_idx], all_y[split_idx:]

y_train_onehot = np.zeros((len(y_train), 5))
y_train_onehot[np.arange(len(y_train)), y_train] = 1

print(f"\nTreningowe: {len(X_train)}, Testowe: {len(X_test)}")

print("\n" + "="*60)
print("TRAINING (wiecej neuronow + wagi klas + wiecej epoch)")
print("="*60)

model = ImprovedNN(24, 24, 5, class_weights=class_weights, learning_rate=0.05)
model.train(X_train, y_train_onehot, epochs=400)

predictions = model.predict(X_test)

correct = 0
results_by_group = {i: {'correct': 0, 'total': 0} for i in range(5)}

for i, pred in enumerate(predictions):
    true_label = y_test[i]
    results_by_group[true_label]['total'] += 1
    if pred == true_label:
        correct += 1
        results_by_group[true_label]['correct'] += 1

group_names = ['A', 'B', 'C', 'D', 'E']
print("\nWYNIKI WG GRUP:")
for i in range(5):
    res = results_by_group[i]
    acc = res['correct'] / res['total'] * 100 if res['total'] > 0 else 0
    print(f"Grupa {group_names[i]}: {res['correct']}/{res['total']} ({acc:.1f}%)")

overall_accuracy = correct / len(y_test) * 100
print(f"\nOGOLNA DOKLADNOSC: {overall_accuracy:.2f}%")

save_path = os.path.join(base_path, "neural_net_improved.pkl")
with open(save_path, "wb") as f:
    pickle.dump(model, f)
print(f"\nZapisano model do: {save_path}")