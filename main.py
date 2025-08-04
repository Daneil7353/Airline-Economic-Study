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


# Ordenar las rutas por efficiency_score para mostrar las mejores al principio
route_scores_sorted = route_scores.sort_values('efficiency_score')

# Gráfico de barras de las mejores rutas por 'efficiency_score'
plt.figure(figsize=(10, 6))
route_scores_sorted['efficiency_score'].head(10).plot(kind='barh', color='skyblue')
plt.title('Top 10 Rutas más Convenientes (Eficiencia vs. Precio)')
plt.xlabel('Efficiency Score (Menor = Mejor)')
plt.ylabel('Ruta')
plt.gca().invert_yaxis()  # Invertir el eje Y para mostrar las mejores rutas en la parte superior
plt.tight_layout()
plt.show()

# Gráfico de barras de precio por ruta
plt.figure(figsize=(10, 6))
route_scores_sorted['price'].head(10).plot(kind='barh', color='salmon')
plt.title('Top 10 Rutas más Baratas (Precio Promedio)')
plt.xlabel('Precio Promedio')
plt.ylabel('Ruta')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
df.groupby('stops_num')['efficiency_score'].mean().plot(kind='line', marker='o', color='teal')
plt.title('Evolución del Efficiency Score según el Número de Escalas')
plt.xlabel('Número de Escalas')
plt.ylabel('Efficiency Score')
plt.grid(True)
plt.tight_layout()
plt.show()

ordered_times = ['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night']
df['departure_time_ordered'] = pd.Categorical(df['departure_time'], categories=ordered_times, ordered=True)
plt.figure(figsize=(10, 6))
df.groupby('departure_time_ordered')['efficiency_score'].mean().plot(kind='line', marker='o', color='orange')
plt.title('Evolución del Efficiency Score según la Hora de Salida')
plt.xlabel('Hora de Salida')
plt.ylabel('Efficiency Score')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
df.groupby('days_left')['efficiency_score'].mean().plot(kind='line', marker='o', color='green')
plt.title('Evolución del Efficiency Score según Días de Antelación')
plt.xlabel('Días de Antelación')
plt.ylabel('Efficiency Score')
plt.grid(True)
plt.tight_layout()
plt.show()