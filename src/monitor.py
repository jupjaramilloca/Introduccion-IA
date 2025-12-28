import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os

# 1. Configuración de API con Cache y Reintentos
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Coordenadas de Rionegro
LAT, LON = 6.1239, -75.3766

url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": LAT, "longitude": LON,
    "hourly": ["temperature_2m", "surface_pressure", "rain", "cloud_cover", "relative_humidity_2m", "wind_speed_10m"],
    "timezone": "auto", "forecast_days": 2
}
responses = openmeteo.weather_api(url, params=params)
response = responses[0] 

# --- DATOS DINÁMICOS ---
elevacion_api = response.Elevation()
lat_api = round(response.Latitude(), 4)
lon_api = round(response.Longitude(), 4)
timezone_api = response.Timezone()

# 2. Procesamiento de Datos
res_hourly = response.Hourly()
df = pd.DataFrame({
    "date": pd.date_range(
        start=pd.to_datetime(res_hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(res_hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=res_hourly.Interval()), inclusive="left"
    ),
    "temp": res_hourly.Variables(0).ValuesAsNumpy(),
    "pressure": res_hourly.Variables(1).ValuesAsNumpy(),
    "rain": res_hourly.Variables(2).ValuesAsNumpy(),
    "clouds": res_hourly.Variables(3).ValuesAsNumpy(),
    "humidity": res_hourly.Variables(4).ValuesAsNumpy(),
    "wind": res_hourly.Variables(5).ValuesAsNumpy()
})

# --- LÓGICA DE PROBABILIDAD (ESTILO ARIMA/SUAVIZADO) ---
# 1. Base bruta
df['prob_bruta'] = ((df['humidity'] * 0.3) + (df['clouds'] * 0.3) + (df['rain'].apply(lambda x: 40 if x > 0 else 0))).clip(0, 100)

# 2. Suavizado (Media móvil de 3 horas para evitar saltos locos)
df['prob_suave'] = df['prob_bruta'].rolling(window=3, center=True).mean().fillna(df['prob_bruta'])

# 3. Factor Presión (Inestabilidad)
presion_normal = 1013
df['factor_presion'] = df['pressure'].apply(lambda x: 1.2 if x < presion_normal else 0.8)

# 4. Probabilidad Final (La que usaremos en el gráfico)
df['prob_lluvia'] = (df['prob_suave'] * df['factor_presion']).clip(5, 95)

# 3. Variables de tiempo
ahora = pd.Timestamp.now(tz='UTC')
fin_prediccion = ahora + pd.Timedelta(hours=6)

# 4. Generación del Tablero
fig = make_subplots(
    rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.12,
    subplot_titles=(
        "🎯 ¿VA A LLOVER? (Probabilidad Calculada)",
        "🌡️ ESTADO DEL AIRE (Temperatura y Humedad)",
        "📉 EL BARÓMETRO (Presión Atmosférica)",
        "☁️ EL CIELO (Nubes y Viento)"
    )
)

# Agregar trazos
fig.add_trace(go.Scatter(x=df['date'], y=df['prob_lluvia'], fill='tozeroy', name="Probabilidad %", line_color='cyan'), row=1, col=1)
fig.add_trace(go.Scatter(x=df['date'], y=df['temp'], name="Calorcito (°C)", line_color='orange'), row=2, col=1)
fig.add_trace(go.Scatter(x=df['date'], y=df['humidity'], name="Vapor/Humedad (%)", line_color='blue'), row=2, col=1)
fig.add_trace(go.Scatter(x=df['date'], y=df['pressure'], name="Peso Aire (hPa)", line_color='red'), row=3, col=1)
fig.add_trace(go.Bar(x=df['date'], y=df['clouds'], name="Nubes (%)", marker_color='gray', opacity=0.4), row=4, col=1)
fig.add_trace(go.Scatter(x=df['date'], y=df['wind'], name="Viento (km/h)", line_color='white'), row=4, col=1)

# Anotaciones "Coquito" y Sombreado
explicaciones = [
    (1, "📊 <b>Probabilidad:</b> Si la montaña cian sube, el riesgo aumenta. Nunca llega a 100% por incertidumbre."),
    (2, "🌡️ <b>Aire:</b> Temp naranja y Humedad azul. Si se cruzan (Temp baja, Hum sube), prepárate."),
    (3, "📉 <b>Presión:</b> Si la línea roja cae por debajo de 1013, el aire está inestable."),
    (4, "☁️ <b>Cielo:</b> Barras grises son nubes. Línea blanca es el viento empujando nubes.")
]

for row, txt in explicaciones:
    y_ref = "y domain" if row == 1 else f"y{row} domain"
    x_ref = "x domain" if row == 1 else f"x{row} domain"
    
    fig.add_annotation(xref=x_ref, yref=y_ref, x=0, y=-0.3, text=txt, showarrow=False, font=dict(size=12, color="lightgray"), align="left")
    
    # Marcador de "Ahora" y Sombreado de predicción (6h)
    fig.add_vline(x=ahora, line_dash="dash", line_color="yellow", row=row, col=1)
    fig.add_vrect(x0=ahora, x1=fin_prediccion, fillcolor="white", opacity=0.1, layer="below", line_width=0, row=row, col=1)

# 5. TÍTULO DINÁMICO
fig.update_layout(
    height=1400, template="plotly_dark",
    title=f"<b>ESTACIÓN METEOROLÓGICA INTELIGENTE</b><br>" +
          f"<span style='font-size:16px; color:cyan;'>📍 Ubicación: {lat_api}°N, {lon_api}°E | " +
          f"⛰️ Elevación: {elevacion_api}| 🌍 Zona: {timezone_api}</span>",
    margin=dict(t=120, b=100),
    showlegend=False
)

fig.write_html("index.html")
print(f"Tablero generado para {lat_api}, {lon_api} (Elev: {elevacion_api}m)")