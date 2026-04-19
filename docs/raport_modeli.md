# Raport z Testow Modeli Klimatycznych

**Data:** 2026-04-19  
**Zbior danych:** folder klimatyczne (27 folderow, 3987 probek)  
**Podzial:** 70% treningowe / 30% testowe

---

## 1. Neural Net Improved (`neural_net_improved.pkl`)

**Architektura:**
- Warstwa wejsciowa: 24 neurony
- Warstwa ukryta: 24 neurony (ReLU)
- Warstwa wyjsciowa: 5 neurony (Sigmoid)
- Funkcja straty: MSE
- Wagi klas: TAK
- Learning rate: 0.05
- Epochs: 400

**Wyniki:**

| Grupa | Poprawne | Wszystkie | Dokladnosc |
|-------|---------|----------|-----------|
| A (Tropikalny) | 109 | 149 | 73.2% |
| B (Suchy) | 164 | 201 | 81.6% |
| C (Podzwrotnikowy) | 335 | 435 | 77.0% |
| D (Kontynentalny) | 271 | 387 | 70.0% |
| E (Tundra/Polarny) | 20 | 25 | 80.0% |

**OGOLNA: 75.10%**

---

## 2. Neural Net V1 (`neural_net_70_30.pkl`)

**Wyniki:**

| Grupa | Poprawne | Wszystkie | Dokladnosc |
|-------|---------|----------|-----------|
| A (Tropikalny) | 47 | 168 | 28.0% |
| B (Suchy) | 167 | 199 | 83.9% |
| C (Podzwrotnikowy) | 431 | 467 | 92.3% |
| D (Kontynentalny) | 242 | 347 | 69.7% |
| E (Tundra/Polarny) | 9 | 16 | 56.2% |

**OGOLNA: 74.85%**

---

## 3. Instar 70/30 V2 (`instar_70_30_v2.pkl`)

Neuronow: 5 (po jednym dla kazdej grupy)

**Wyniki:**

| Grupa | Poprawne | Wszystkie | Dokladnosc |
|-------|---------|----------|-----------|
| A | 5 | 162 | 3.1% |
| B | 188 | 188 | 100.0% |
| C | 172 | 459 | 37.5% |
| D | 5 | 383 | 1.3% |
| E | 0 | 21 | 0.0% |

**OGOLNA: 30.50%**

---

## 4. Instar 25/75 V2 (`instar_25_75_v2.pkl`)

Neuronow: ~25 (5 na grupe)

**Wyniki:**

| Grupa | Poprawne | Wszystkie | Dokladnosc |
|-------|---------|----------|-----------|
| A | 124 | 402 | 30.8% |
| B | 120 | 467 | 25.7% |
| C | 329 | 1137 | 28.9% |
| D | 139 | 929 | 15.0% |
| E | 36 | 58 | 62.1% |

**OGOLNA: 24.99%**

---

## 5. Instar 01 / 02 (`instar01.pkl`, `instar02.pkl`)

Neuronow: 32 (system 16 stref x 2 polkule)

**Wyniki:**

| Klimat | Poprawne | Wszystkie | Dokladnosc |
|-------|---------|----------|-----------|
| Rownikowy | 355 | 535 | 66.4% |
| Zwrotnikowy | 77 | 622 | 12.4% |
| Podzwrotnikowy | 205 | 1515 | 13.5% |
| Monsunowy | 4 | 117 | 3.4% |
| Kontynentalny | 53 | 1121 | 4.7% |
| Tundra/Polarny | 12 | 77 | 15.6% |

**OGOLNA: 17.71%**

---

## 6. Instar CO4 (`instarco4.pkl`)

**Wyniki:**

| Klimat | Poprawne | Wszystkie | Dokladnosc |
|-------|---------|----------|-----------|
| Rownikowy | 124 | 535 | 23.2% |
| Zwrotnikowy | 67 | 622 | 10.8% |
| Podzwrotnikowy | 157 | 1515 | 10.4% |
| Monsunowy | 12 | 117 | 10.3% |
| Kontynentalny | 84 | 1121 | 7.5% |
| Tundra/Polarny | 7 | 77 | 9.1% |

**OGOLNA: 11.31%**

---

## Podsumowanie

### Najlepsze Modele (grupowanie A,B,C,D,E)

| Model | Dokladnosc |
|-------|-----------|
| **Neural Net Improved** | **75.10%** |
| Neural Net v1 | 74.85% |
| Instar 70/30 v2 | 30.50% |
| Instar 25/75 v2 | 24.99% |

### Modele Oryginalne (system 16 stref)

| Model | Dokladnosc |
|-------|-----------|
| Instar 01/02 | 17.71% |
| Instar co4 | 11.31% |

---

## Wnioski

1. **Neural Net Improved** osiąga najwyższa dokladnosc (75.10%) dzieki waznym klas

2. Grupy A i E sa najtrudniejsze - za malo danych treningowych

3. Instar nie nadaje sie do tego zadania

4. Modele instar01/02 sa identyczne