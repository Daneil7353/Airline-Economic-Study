import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

df = pd.read_csv('data/clean_data/airlines_flights_data_clean.csv')

# Mostrar las clases disponibles
print(df.value_counts())

# Filtrar solo clase Economy
df_econ = df[df['class'].str.lower() == 'economy'].copy()

# Delimitar precios muy dispares para eliminar valores que distorsionen la vista real
q_low, q_high = df_econ['price'].quantile([0.01, 0.99])
df_econ = df_econ[(df_econ['price'] >= q_low) & (df_econ['price'] <= q_high)]

stops_map = {
    'zero': 0,
    'one': 1,
    'two': 2,
    'three': 3
}
df['stops_num'] = df['stops'].map(stops_map)

time_map = {
    'Early_Morning': 1,
    'Morning': 2,
    'Afternoon': 2,
    'Evening': 1,
    'Night': 0,
    'Late_Night': 0
}
df['time_score'] = df['departure_time'].map(time_map)

def duration_to_minutes(duration_str):
    hours = 0
    minutes = 0
    h_match = re.search(r'(\d+)h', duration_str)
    m_match = re.search(r'(\d+)m', duration_str)
    if h_match:
        hours = int(h_match.group(1))
    if m_match:
        minutes = int(m_match.group(1))
    return hours * 60 + minutes

df['duration'] = df['duration'].fillna('').astype(str)
df['duration_min']=df['duration'].apply(duration_to_minutes)

df['route'] = df['source_city'] + ' → ' + df['destination_city']

df.dropna(subset=['price', 'stops_num', 'time_score', 'duration_min'], inplace=True)

# Normalizar todas las variables con NumPy
def normalize(series):
    min_val = series.min()
    max_val = series.max()
    return (series - min_val) / (max_val - min_val) if max_val != min_val else 0

df['price_norm']     = normalize(df['price'])
df['stops_norm']     = normalize(df['stops_num'])
df['time_norm']      = normalize(df['time_score'])
df['duration_norm']  = normalize(df['duration_min'])

# Ajusta los pesos según tus preferencias
df['efficiency_score'] = (
    0.7 * df['price_norm'] +
    0.1 * df['stops_norm'] +
    0.1 * (1 - df['time_norm']) + # cuanto más tarde, peor
    0.1 * df['duration_norm']
)


route_scores = df.groupby('route').agg({
    'efficiency_score': 'mean',
    'price': 'mean'
}).sort_values('efficiency_score', ascending=1)

