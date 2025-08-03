import pandas as pd
import numpy as np
import matplotlib as plot

df = pd.read_csv('data/clean_data/airlines_flights_data_clean.csv')

# Mostrar las clases disponibles
print(df.value_counts())

# Filtrar solo clase Economy
df_econ = df[df['class'].str.lower() == 'economy'].copy()

# Opcional: eliminar precios atípicos extremos (por ejemplo, outliers superiores)
q_low, q_high = df_econ['price'].quantile([0.01, 0.99])
df_econ = df_econ[(df_econ['price'] >= q_low) & (df_econ['price'] <= q_high)]
