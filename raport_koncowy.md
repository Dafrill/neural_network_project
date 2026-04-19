# Projekt Sieci Neuronowe - Klasyfikacja Klimatow Koppen

**Autorzy:** Magdalena Tałaj, Natalia Nowak, Viktoria Toman

---

## 1. Opis Problemu

### 1.1 Cel Projektu

Celem projektu byla implementacja i analiza sieci neuronowych do automatycznej klasyfikacji klimatow na podstawie danych meteorologicznych (temperatura i opady) zgodnych z klasyfikacja Koppen-Geiger.

### 1.2 Dane

- **Zrodlo:** Dane ze stacji meteorologicznych z całego świata z serwisu meteostat
- **Liczba probek:** 9504 rekordow
- **Cecha wejsciowe:** 24 wartosci (12 temperatur miesiecznych + 12 opadow miesiecznych)
- **Klasy wyjsciowe:** 5 grup klimatycznych (A, B, C, D, E):
  - **A** (Tropikalny): 535 probek
  - **B** (Suchy): 622 probek
  - **C** (Podzwrotnikowy): 1515 probek
  - **D** (Kontynentalny): 1238 probek
  - **E** (Tundra/Polarny): 77 probek

### 1.3 Problem

Niezbalansowanie klas - klasa E ma tylko 77 probek vs 1515 dla klasy C.

---

## 2. Przeglad Literatury

"Sieci neuronowe", Ryszard Tadeusiewicz

### 2.1 Klasyfikacja Koppen-Geiger

Klasyfikacja klimatow Koppen-Geiger jest jedyna z najczestszych systematyk klimatycznych na swiecie. Wedlug klasycznej implementacji [1]:

- **A** - Klimat tropikalny (temp. srednia >= 18°C)
- **B** - Klimat suchy (niskie opady)
- **C** - Klimat podzwrotnikowy (umiarkowane zimy)
- **D** - Klimat kontynentalny (zimne zimy)
- **E** - Klimat polarny (temp. < 10°C)

[1] Kottek, M., et al. (2006). "World Maps of the Koppen-Geiger Climate Classification." Meteorologische Zeitschrift.

### 2.2 Sieci Neuronowe do Klasyfikacji

Sieci typu feed-forward z algorytmem wstecznej propagacji bledu (BP) sa standardowym narzedziem do klasyfikacji wzorcow [2]:

- Warstwy ukryte z funkcja ReLU/Sigmoid
- Funkcja straty: cross-entropy lub MSE
- Optymalizacja: gradient descent

[2] Haykin, S. (2009). "Neural Networks and Learning Machines." Pearson.

### 2.3 Instar (Competitive Learning)

Sieci Instar (Instar Algorithm) sa jednym z pierwszych modeli nienadzorowanych [3]. Uzywaja konkurencji pomiedzy neuronami i aktualizacji tylko zwyciezcy.

[3] Kohonen, T. (1982). "Self-Organized Formation of Topologically Correct Feature Maps." Biological Cybernetics.

---

## 3. Eksperymenty i Wyniki

Kazdy eksperyment powtorzono 5 razy z roznymi ziarnami losowymi. Podane sa wartosci srednie.

### 3.1 Wplyw Liczby Neuronow w Warstwie Ukrytej

| Neurony | Dokladnosc (srednia) | Odchylenie |
|--------|---------------------|------------|
| 4 | 71.23% | 1.85% |
| 8 | 73.47% | 3.34% |
| 16 | 63.36% | 21.61% |
| **24** | **75.79%** | 1.12% |
| 32 | 75.49% | 2.94% |

**Wniosek:** 24 neurony daja najlepsze i najbardziej stabilne wyniki.

### 3.2 Wplyw Liczby Warstw

| Konfiguracja | Dokladnosc |
|-------------|-----------|
| 1 warstwa ukryta (8 neuronow) | 69.29% |
| 1 warstwa ukryta (16 neuronow) | 72.60% |
| **1 warstwa ukryta (24 neuronow)** | **75.94%** |

**Wniosek:** Jedna warstwa ukryta wystarcza dla tego problemu.

### 3.3 Wplyw Funkcji Aktywacji

| Funkcja | Dokladnosc |
|--------|-----------|
| Sigmoid | 21.19% |
| Tanh | 36.36% |
| **ReLU** | **72.60%** |

**Wniosek:** ReLU znacznie przewyzsza sigmoid i tanh.

### 3.4 Wplyw Learning Rate

| LR | Dokladnosc |
|----|-----------|
| 0.01 | 68.20% |
| **0.05** | **73.92%** |
| 0.1 | 55.17% |
| 0.5 | 39.67% |

**Wniosek:** Zbyt duzy LR powoduje niestabilnosc uczenia.

### 3.5 Wplyw Podzialu Zbiorow

| Treningowe | Testowe | Dokladnosc |
|-----------|---------|-----------|
| 50% | 50% | 38.53% |
| 60% | 40% | 61.81% |
| **70%** | **30%** | **55.17%** |
| 80% | 20% | 55.06% |
| 90% | 10% | 49.27% |

**Wniosek:** 70% jest optymalnym kompromisem.

### 3.6 Porownanie Modeli

| Model | Neurony | Funkcja aktywacji | LR | Epochs | Dokladnosc |
|-------|--------|------------------|-----|-------|-----------|
| Neural Net v1 | 16 | ReLU -> Sigmoid | 0.1 | 200 | **74.85%** |
| Neural Net Improved | 24 | ReLU -> Sigmoid | 0.05 | 400 | **75.10%** |
| Instar 70/30 | 5 | brak | 0.2 | - | 30.50% |
| Instar 25/75 | ~25 | brak | 0.1 | - | 24.99% |
| Instar 01/02 | 32 | brak | 0.2 | - | 17.71% |

### 3.7 Wyniki Ostatecznego Modelu (Neural Net Improved)

Najlepszy model: Neural Net Improved z wagami klas

| Grupa | Poprawne | Wszystkie | Dokladnosc |
|-------|---------|----------|-----------|
| A (Tropikalny) | 109 | 149 | **73.2%** |
| B (Suchy) | 164 | 201 | **81.6%** |
| C (Podzwrotnikowy) | 335 | 435 | **77.0%** |
| D (Kontynentalny) | 271 | 387 | **70.0%** |
| E (Tundra/Polarny) | 20 | 25 | **80.0%** |
| **OGOLNA** | **899** | **1197** | **75.10%** |

---

## 4. Architektura Najlepszego Modelu

```
Wejscie (24) -> ReLU -> Warstwa ukryta (24) -> Sigmoid -> Wyjscie (5 klasy)
```

- **Funkcja straty:** MSE
- **Wagi klas:** TAK (dla niezbalansowanych danych)
- **Learning rate:** 0.05
- **Epochs:** 400
- **Inicjalizacja:** Xavier

---

## 5. Wnioski

1. **Najlepszy model osiąga 75.10% dokładności** przy uzyciu sieci feed-forward z ReLU + Sigmoid.

2. **Kluczowe parametry:**
   - Liczba neuronow: 24
   - Funkcja aktywacji: ReLU
   - Learning rate: 0.05
   - Wagi klas dla niezbalansowanych danych

3. **Sieci Instar bez funkcji aktywacji nie nadaja sie** do tego zadania (max 30%).

4. **Grupy A i E** sa najtrudniejsze do rozpoznania z powodu malej liczby probek.

5. **Dane wymagaja lepszego przygotowania** - wiecej probek dla grup z malej liczba.

---

## 6. Zrodla

1. Kottek, M., et al. (2006). World Maps of the Koppen-Geiger Climate Classification. Meteorologische Zeitschrift, 15(3), 259-263.

2. Haykin, S. (2009). Neural Networks and Learning Machines (3rd ed.). Pearson.

3. Kohonen, T. (1982). Self-Organized Formation of Topologically Correct Feature Maps. Biological Cybernetics, 43(1), 59-69.

4. LeCun, Y., et al. (1998). Gradient-Based Learning Applied to Document Recognition. Proceedings of the IEEE, 86(11), 2278-2324.

---


