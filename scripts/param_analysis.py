import numpy as np
import os
import pandas as pd
import pickle
import random
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from network_functions import GetX_Vector

random.seed(42)
np.random.seed(42)

class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.lr = learning_rate
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


base_path = os.path.join(os.path.dirname(__file__), '..')
klimatyczne_path = os.path.join(base_path, "data", "klimatyczne")

climate_folders = [d for d in os.listdir(klimatyczne_path) if os.path.isdir(os.path.join(klimatyczne_path, d))]

all_X = []
all_y = []

for folder in climate_folders:
    folder_path = os.path.join(klimatyczne_path, folder)
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
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

print(f"Dane: {len(all_X)} probek")

results = []

print("\n" + "="*60)
print("1. WPŁYW LICZBY NEURONÓW W WARSTWIE UKRYTEJ")
print("="*60)

for n_neurons in [4, 8, 16, 24, 32]:
    accuracies = []
    for run in range(5):
        np.random.seed(run*100)
        indices = np.arange(len(all_X))
        np.random.shuffle(indices)
        all_X = all_X[indices]
        all_y = all_y[indices]
        
        split_idx = int(len(all_X) * 0.7)
        X_train, X_test = all_X[:split_idx], all_X[split_idx:]
        y_train, y_test = all_y[:split_idx], all_y[split_idx:]
        
        y_train_onehot = np.zeros((len(y_train), 5))
        y_train_onehot[np.arange(len(y_train)), y_train] = 1
        
        model = SimpleNN(24, n_neurons, 5, learning_rate=0.1)
        model.train(X_train, y_train_onehot, epochs=200)
        
        preds = model.predict(X_test)
        acc = np.mean(preds == y_test) * 100
        accuracies.append(acc)
    
    avg = np.mean(accuracies)
    std = np.std(accuracies)
    results.append(f"  {n_neurons} neuronow: {avg:.2f}% (+/- {std:.2f}%)")
    print(f"  {n_neurons} neuronow: {avg:.2f}% (+/- {std:.2f}%)")

print("\n" + "="*60)
print("2. WPŁYW LICZBY WARSTW")
print("="*60)

three_layer_results = []
for n_neurons in [8, 16, 24]:
    accuracies = []
    for run in range(5):
        np.random.seed(run*100)
        indices = np.arange(len(all_X))
        np.random.shuffle(indices)
        all_X_s = all_X[indices]
        all_y_s = all_y[indices]
        
        split_idx = int(len(all_X) * 0.7)
        X_train, X_test = all_X_s[:split_idx], all_X_s[split_idx:]
        y_train, y_test = all_y_s[:split_idx], all_y_s[split_idx:]
        
        y_train_onehot = np.zeros((len(y_train), 5))
        y_train_onehot[np.arange(len(y_train)), y_train] = 1
        
        np.random.seed(run*100)
        W1 = np.random.randn(24, n_neurons) * 0.1
        b1 = np.zeros((1, n_neurons))
        W2 = np.random.randn(n_neurons, 5) * 0.1
        b2 = np.zeros((1, 5))
        
        for epoch in range(200):
            z1 = np.dot(X_train, W1) + b1
            a1 = np.maximum(0, z1)
            z2 = np.dot(a1, W2) + b2
            a2 = 1 / (1 + np.exp(-np.clip(z2, -500, 500)))
            
            delta2 = (a2 - y_train_onehot) * a2 * (1 - a2)
            dW2 = np.dot(a1.T, delta2) / len(y_train)
            db2 = np.sum(delta2, axis=0, keepdims=True) / len(y_train)
            delta1 = np.dot(delta2, W2.T) * (a1 > 0)
            dW1 = np.dot(X_train.T, delta1) / len(y_train)
            db1 = np.sum(delta1, axis=0, keepdims=True) / len(y_train)
            W2 -= 0.1 * dW2
            b2 -= 0.1 * db2
            W1 -= 0.1 * dW1
            b1 -= 0.1 * db1
        
        z1 = np.dot(X_test, W1) + b1
        a1 = np.maximum(0, z1)
        z2 = np.dot(a1, W2) + b2
        a2 = 1 / (1 + np.exp(-np.clip(z2, -500, 500)))
        preds = np.argmax(a2, axis=1)
        acc = np.mean(preds == y_test) * 100
        accuracies.append(acc)
    
    avg = np.mean(accuracies)
    results.append(f"  2 warstwy ({n_neurons} neuronow): {avg:.2f}%")
    print(f"  2 warstwy ({n_neurons} neuronow): {avg:.2f}%")

print("\n" + "="*60)
print("3. WPŁYW FUNKCJI AKTYWACJI")
print("="*60)

for activation in ['sigmoid', 'tanh', 'relu']:
    accuracies = []
    for run in range(5):
        np.random.seed(run*100)
        indices = np.arange(len(all_X))
        np.random.shuffle(indices)
        all_X_s = all_X[indices]
        all_y_s = all_y[indices]
        
        split_idx = int(len(all_X) * 0.7)
        X_train, X_test = all_X_s[:split_idx], all_X_s[split_idx:]
        y_train, y_test = all_y_s[:split_idx], all_y_s[split_idx:]
        
        y_train_onehot = np.zeros((len(y_train), 5))
        y_train_onehot[np.arange(len(y_train)), y_train] = 1
        
        np.random.seed(run*100)
        W1 = np.random.randn(24, 16) * 0.1
        b1 = np.zeros((1, 16))
        W2 = np.random.randn(16, 5) * 0.1
        b2 = np.zeros((1, 5))
        
        for epoch in range(200):
            z1 = np.dot(X_train, W1) + b1
            if activation == 'sigmoid':
                a1 = 1 / (1 + np.exp(-z1))
            elif activation == 'tanh':
                a1 = np.tanh(z1)
            else:
                a1 = np.maximum(0, z1)
            
            z2 = np.dot(a1, W2) + b2
            a2 = 1 / (1 + np.exp(-np.clip(z2, -500, 500)))
            
            delta2 = (a2 - y_train_onehot) * a2 * (1 - a2)
            dW2 = np.dot(a1.T, delta2) / len(y_train)
            db2 = np.sum(delta2, axis=0, keepdims=True) / len(y_train)
            delta1 = np.dot(delta2, W2.T) * (a1 > 0)
            dW1 = np.dot(X_train.T, delta1) / len(y_train)
            db1 = np.sum(delta1, axis=0, keepdims=True) / len(y_train)
            W2 -= 0.1 * dW2
            b2 -= 0.1 * db2
            W1 -= 0.1 * dW1
            b1 -= 0.1 * db1
        
        z1 = np.dot(X_test, W1) + b1
        a1 = np.maximum(0, z1)
        z2 = np.dot(a1, W2) + b2
        a2 = 1 / (1 + np.exp(-np.clip(z2, -500, 500)))
        preds = np.argmax(a2, axis=1)
        acc = np.mean(preds == y_test) * 100
        accuracies.append(acc)
    
    avg = np.mean(accuracies)
    results.append(f"  {activation}: {avg:.2f}%")
    print(f"  {activation}: {avg:.2f}%")

print("\n" + "="*60)
print("4. WPŁYW LEARNING RATE")
print("="*60)

for lr in [0.01, 0.05, 0.1, 0.5]:
    accuracies = []
    for run in range(5):
        np.random.seed(run*100)
        indices = np.arange(len(all_X))
        np.random.shuffle(indices)
        all_X_s = all_X[indices]
        all_y_s = all_y[indices]
        
        split_idx = int(len(all_X) * 0.7)
        X_train, X_test = all_X_s[:split_idx], all_X_s[split_idx:]
        y_train, y_test = all_y_s[:split_idx], all_y_s[split_idx:]
        
        y_train_onehot = np.zeros((len(y_train), 5))
        y_train_onehot[np.arange(len(y_train)), y_train] = 1
        
        np.random.seed(run*100)
        model = SimpleNN(24, 16, 5, learning_rate=lr)
        model.train(X_train, y_train_onehot, epochs=200)
        
        preds = model.predict(X_test)
        acc = np.mean(preds == y_test) * 100
        accuracies.append(acc)
    
    avg = np.mean(accuracies)
    results.append(f"  lr={lr}: {avg:.2f}%")
    print(f"  lr={lr}: {avg:.2f}%")

print("\n" + "="*60)
print("5. WPŁYW WIELKOŚCI ZBIORU TRENINGOWEGO")
print("="*60)

for ratio in [0.5, 0.6, 0.7, 0.8, 0.9]:
    accuracies = []
    for run in range(5):
        np.random.seed(run*100)
        indices = np.arange(len(all_X))
        np.random.shuffle(indices)
        all_X_s = all_X[indices]
        all_y_s = all_y[indices]
        
        split_idx = int(len(all_X) * ratio)
        X_train, X_test = all_X_s[:split_idx], all_X_s[split_idx:]
        y_train, y_test = all_y_s[:split_idx], all_y_s[split_idx:]
        
        y_train_onehot = np.zeros((len(y_train), 5))
        y_train_onehot[np.arange(len(y_train)), y_train] = 1
        
        np.random.seed(run*100)
        model = SimpleNN(24, 16, 5, learning_rate=0.1)
        model.train(X_train, y_train_onehot, epochs=200)
        
        preds = model.predict(X_test)
        acc = np.mean(preds == y_test) * 100
        accuracies.append(acc)
    
    avg = np.mean(accuracies)
    results.append(f"  {int(ratio*100)}% treningowy: {avg:.2f}%")
    print(f"  {int(ratio*100)}% treningowy: {avg:.2f}%")

print("\nZakonczono analize.")