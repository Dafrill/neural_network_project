def GetX_Vector(df):
    # 0-11: temp 
    # 12-33: prcp
    
    temps = df['temp'].tolist()
    prcps = df['prcp'].tolist()

    # Łączymy listy (12 elementów temp + 12 elementów prcp = 24)
    x_vector = temps + prcps

    # Zwracamy jako słownik, aby pasowało do Twojej funkcji Visualize
    return x_vector
import matplotlib.pyplot as plt
import numpy as np


import matplotlib.pyplot as plt

def Visualize(weights, labels=None):
    """
    weights: macierz wag (np. z Twojej sieci Instar)
    labels: opcjonalna lista nazw (np. ['Równikowy', 'Polarny', ...])
    """
    for i, vector in enumerate(weights):
        
        name = labels[i] if labels is not None else f"Neuron/Wzorzec nr {i}"
        
        
        temps = vector[0:12]
        prcps = vector[12:24]
        months = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII']
        
        fig, ax1 = plt.subplots(figsize=(10, 5))
        
        
        max_p = max(prcps) if max(prcps) > 0 else 100
        prcp_limit = max_p + 50 
        ax1.set_ymargin(0) 
        ax1.set_ylim(-prcp_limit, prcp_limit)
        
        ax1.bar(months, prcps, color='skyblue', label='Opady', alpha=0.7)
        ax1.set_ylabel('Opady (mm)', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        
        
        ax2 = ax1.twinx()
        ax2.set_ymargin(0) 
        ax2.set_ylim(-40, 40)
        
        ax2.plot(months, temps, color='red', marker='o', linewidth=2, label='Temperatura')
        ax2.set_ylabel('Temperatura (°C)', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        
       
        ax1.axhline(0, color='black', linewidth=1.2, linestyle='--') 
        
        
        ax1.set_yticks([val for val in ax1.get_yticks() if val >= 0])

        plt.title(f'Wizualizacja Wzorca: {name}')
        fig.tight_layout()
        plt.show()

all_climates = {
    # --- 1. RÓWNIKOWY (Brak dużych różnic między półkulami) ---
    # Stale wysoka temperatura (~27°C) i potężne opady przez cały rok
    "1N. Równikowy wilgotny (Wzorzec)": 
        [27, 27, 27, 28, 28, 27, 27, 27, 28, 28, 27, 27] + [250, 200, 210, 200, 190, 180, 170, 180, 190, 220, 250, 300],

    "1S. Równikowy wilgotny (Wzorzec)": 
        [27, 27, 27, 27, 27, 26, 26, 27, 28, 28, 28, 27] + [300, 250, 220, 190, 180, 170, 180, 170, 190, 200, 210, 250],
    # --- 2. PODRÓWNIKOWY WILGOTNY ---
    "2N. Podrównikowy wilg. (N)": [25, 26, 27, 28, 28, 27, 26, 26, 26, 26, 26, 25] + [10, 15, 40, 110, 180, 250, 280, 260, 220, 150, 40, 10],
    "2S. Podrównikowy wilg. (Cairns)": [28, 28, 27, 26, 24, 23, 22, 23, 24, 26, 27, 28] + [380, 350, 320, 150, 60, 40, 30, 30, 40, 70, 130, 220],

    # --- 3. PODRÓWNIKOWY SUCHY ---
    "3N. Podrównikowy suchy (Niamey)": [24, 27, 31, 34, 34, 32, 29, 28, 29, 31, 28, 25] + [0, 0, 5, 10, 35, 75, 150, 170, 80, 15, 0, 0],
    "3S. Podrównikowy suchy (Darwin)": [28, 28, 28, 28, 27, 25, 25, 26, 28, 29, 29, 29] + [300, 250, 200, 50, 5, 1, 1, 5, 15, 70, 140, 250],

    # --- 4. ZWROTNIKOWY WILGOTNY ---
    "4N. Zwrotnikowy wilg. (Miami)": [20, 21, 23, 24, 27, 28, 29, 29, 28, 27, 24, 21] + [70, 50, 70, 80, 130, 220, 160, 220, 250, 160, 80, 50],
    "4S. Zwrotnikowy wilg. (Sao Paulo)": [23, 23, 22, 21, 18, 17, 16, 17, 18, 20, 21, 22] + [240, 220, 160, 80, 70, 60, 50, 40, 80, 130, 150, 200],

    # --- 5. ZWROTNIKOWY SUCHY ---
    "5N. Zwrotnikowy suchy (Rijad)": [15, 18, 23, 29, 34, 37, 39, 39, 35, 30, 22, 16] + [15, 20, 25, 25, 10, 0, 0, 0, 0, 1, 10, 15],
    "5S. Zwrotnikowy suchy (Alice Springs)": [29, 28, 25, 20, 15, 12, 12, 14, 19, 23, 26, 28] + [40, 40, 30, 15, 15, 15, 15, 10, 10, 20, 30, 40],

    # --- 6. PODZWROTNIKOWY MORSKI (Śródziemnomorski) ---
    "6N. Podzwrotnikowy morski (Rzym)": [8, 9, 12, 14, 19, 23, 26, 26, 22, 18, 13, 9] + [80, 70, 65, 60, 50, 30, 15, 25, 70, 95, 110, 100],
    "6S. Podzwrotnikowy morski (Cape Town)": [21, 21, 20, 18, 15, 13, 12, 13, 14, 17, 18, 20] + [15, 15, 20, 50, 90, 110, 100, 90, 50, 40, 25, 15],

    # --- 7. PODZWROTNIKOWY KONTYNENTALNY ---
    "7N. Podzwrotnikowy kont. (Taszkent)": [2, 5, 11, 18, 23, 28, 30, 29, 23, 16, 9, 4] + [50, 50, 70, 60, 30, 10, 5, 2, 5, 25, 45, 55],
    "7S. Podzwrotnikowy kont. (Mendoza)": [25, 23, 20, 16, 12, 8, 8, 10, 13, 18, 22, 24] + [30, 30, 30, 15, 10, 10, 10, 10, 15, 20, 25, 30],

    # --- 8. PODZWROTNIKOWY MONSUNOWY ---
    "8N. Podzwrotnikowy mons. (Szanghaj)": [5, 6, 10, 16, 21, 25, 28, 28, 24, 19, 13, 7] + [60, 60, 90, 90, 110, 170, 150, 150, 140, 60, 50, 40],
    "8S. Podzwrotnikowy mons. (Brisbane)": [25, 25, 24, 22, 19, 16, 15, 16, 19, 21, 23, 24] + [150, 160, 140, 90, 70, 70, 50, 40, 40, 70, 100, 130],

    # --- 9. UMIARKOWANY CIEPŁY MORSKI ---
    "9N. Umiarkowany morski (Londyn)": [5, 5, 8, 10, 13, 16, 19, 18, 16, 12, 8, 5] + [55, 40, 40, 45, 45, 45, 45, 50, 50, 65, 60, 55],
    "9S. Umiarkowany morski (Melbourne)": [20, 20, 18, 16, 13, 11, 10, 11, 13, 15, 17, 19] + [45, 45, 50, 55, 55, 50, 50, 50, 60, 65, 60, 60],

    # --- 10. UMIARKOWANY PRZEJŚCIOWY ---
    "10N. Umiarkowany przejściowy (Warszawa)": [-1, 0, 4, 9, 14, 18, 20, 19, 14, 9, 4, 1] + [30, 30, 35, 45, 60, 75, 80, 75, 50, 40, 40, 40],
    "10S. Umiarkowany przejściowy (Dunedin)": [15, 15, 14, 12, 9, 7, 7, 8, 9, 11, 12, 14] + [70, 65, 70, 70, 80, 80, 80, 80, 70, 70, 70, 80],

    # --- 11. UMIARKOWANY CIEPŁY KONTYNENTALNY ---
    "11N. Umiarkowany kont. (Chicago)": [-4, -2, 4, 10, 16, 21, 24, 23, 19, 12, 5, -1] + [50, 50, 65, 90, 95, 100, 95, 100, 80, 80, 80, 60],
    "11S. Umiarkowany kont. (Santa Rosa, ARG)": [24, 23, 20, 15, 11, 8, 7, 10, 13, 17, 21, 23] + [80, 80, 90, 50, 30, 20, 20, 25, 45, 75, 85, 95],

    # --- 12. UMIARKOWANY CHŁODNY MORSKI ---
    "12N. Umiarkowany chł. morski (Bergen)": [2, 2, 4, 7, 11, 13, 15, 15, 12, 9, 5, 3] + [190, 150, 170, 110, 100, 130, 150, 190, 280, 270, 260, 230],
    "12S. Umiarkowany chł. morski (Ushuaia)": [10, 10, 8, 6, 4, 2, 2, 3, 5, 7, 8, 9] + [50, 45, 50, 55, 50, 50, 45, 45, 40, 45, 45, 50],

    # --- 13. UMIARKOWANY CHŁODNY KONTYNENTALNY (Tajga) ---
    "13N. Tajga (Irkuck)": [-18, -14, -5, 4, 12, 18, 20, 18, 11, 2, -9, -16] + [15, 10, 10, 20, 35, 65, 110, 90, 55, 25, 20, 20],
    "13S. Tajga (brak - góry w Patagonii)": [12, 11, 9, 6, 3, 0, -1, 1, 4, 7, 9, 11] + [150, 140, 160, 200, 250, 260, 240, 210, 170, 150, 130, 140],

    # --- 14. SUBPOLARNY (Tundra) ---
    "14N. Subpolarny (Murmańsk)": [-10, -10, -6, -1, 4, 10, 13, 12, 8, 2, -4, -8] + [30, 25, 25, 25, 35, 60, 70, 70, 60, 50, 45, 40],
    "14S. Subpolarny (Stacja Esperance)": [1, 1, 0, -3, -6, -9, -10, -10, -8, -5, -2, 0] + [40, 35, 40, 45, 50, 50, 50, 50, 45, 40, 35, 40],

    # --- 15. POLARNY ---
    "15N. Polarny (Eureka, Kanada)": [-37, -38, -37, -27, -10, 2, 6, 4, -8, -22, -30, -35] + [3, 2, 3, 4, 5, 8, 15, 15, 10, 8, 5, 3],
    "15S. Polarny (Vostok)": [-30, -45, -55, -65, -65, -65, -67, -68, -65, -55, -40, -30] + [1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1],

    # --- 16. GÓRSKI ---
    "16N. Górski (Alpy)": [-4, -3, 0, 3, 7, 11, 13, 13, 10, 6, 0, -3] + [100, 110, 120, 130, 150, 180, 170, 160, 130, 110, 110, 100],
    "16S. Górski (Andy południowe)": [10, 9, 7, 4, 2, -1, -2, -1, 2, 5, 7, 9] + [200, 180, 200, 250, 300, 310, 290, 260, 200, 180, 160, 180]
}

if __name__ == "__main__":
    Visualize(all_climates)