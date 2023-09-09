import pickle

import os
import sys

model = 'model_v1'

with open(f'../{model}.pkl', 'rb') as f:
        rf = pickle.load(f)

# Abre un archivo de texto en modo escritura
with open(f"hiperparametros_{model}.txt", "w", encoding="utf-8") as file:
    # Escribe los hiperparámetros en el archivo
    #file.write(f"Tipo de modelo:{type(rf)}\n")
    for k,v in rf.get_params().items():
        file.write(f"{k}: {v}\n")

def make_prediction(input_data:list):
    new_variables = [input_data]
    return int(rf.predict(new_variables)[0])


# prueba

