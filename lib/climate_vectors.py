import csv
import os
import json

cleaned_dir = '/home/magda/Dokumenty/esi_projekt/neural_network_project/cleaned_data'

climate_vectors = {
    "af": {
        "description": "Wilgotny tropikalny",
        "temps": [
            [27.5, 27.7, 28.7, 29.0, 29.1, 28.7, 29.9, 28.4, 28.3, 28.0, 27.6, 27.3],
            [28.0, 27.9, 27.4, 27.0, 23.4, 22.1, 20.8, 20.8, 22.5, 25.3, 27.1, 27.8]
        ],
        "prcp": [
            [132.0, 107.0, 70.0, 110.0, 152.0, 87.0, 77.0, 209.0, 162.0, 141.0, 98.0, 112.0],
            [261.0, 474.0, 292.0, 184.0, 401.0, 133.0, 296.0, 257.0, 68.0, 102.0, 165.0, 218.0]
        ]
    },
    "am": {
        "description": "Monsunowy",
        "temps": [[], []],
        "prcp": [[], []]
    },
    "aw": {
        "description": "Sawanna",
        "temps": [[], []],
        "prcp": [[], []]
    },
    "cfa": {
        "description": "Wilgotny subtropikalny",
        "temps": [[], []],
        "prcp": [[], []]
    },
    "cfb": {
        "description": "Oceaniczny",
        "temps": [[], []],
        "prcp": [[], []]
    },
    "csa": {
        "description": "Srodziemnomorski goracy",
        "temps": [[], []],
        "prcp": [[], []]
    },
    "csb": {
        "description": "Srodziemnomorski cieply",
        "temps": [[], []],
        "prcp": [[], []]
    },
    "dfb": {
        "description": "Kontynentalny cieply",
        "temps": [[], []],
        "prcp": [[], []]
    },
    "dfc": {
        "description": "Borealny",
        "temps": [[], []],
        "prcp": [[], []]
    },
    "dsb": {
        "description": "Kontynentalny suchy cieply",
        "temps": [[], []],
        "prcp": [[], []]
    },
    "et": {
        "description": "Tundra",
        "temps": [[], []],
        "prcp": [[], []]
    },
    "bwh": {
        "description": "Pustynny goracy",
        "temps": [[], []],
        "prcp": [[], []]
    },
    "bsk": {
        "description": "Stepowy chlodny",
        "temps": [[], []],
        "prcp": [[], []]
    }
}

# AF - 2 stacje (4 wektory: 2x temp + 2x prcp)
climate_vectors["af"]["temps"][0] = [27.5, 27.7, 28.7, 29.0, 29.1, 28.7, 29.9, 28.4, 28.3, 28.0, 27.6, 27.3]
climate_vectors["af"]["prcp"][0] = [132.0, 107.0, 70.0, 110.0, 152.0, 87.0, 77.0, 209.0, 162.0, 141.0, 98.0, 112.0]

print("AF - Wilgotny tropikalny (2 stacje = 4 wektory):")
print("Temps 1:", climate_vectors["af"]["temps"][0])
print("Prcp 1:", climate_vectors["af"]["prcp"][0])
print("Temps 2:", climate_vectors["af"]["temps"][1])
print("Prcp 2:", climate_vectors["af"]["prcp"][1])

# Zapisz do JSON
with open('climate_vectors.json', 'w') as f:
    json.dump(climate_vectors, f, indent=2)