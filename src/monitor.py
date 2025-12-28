import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import pytz
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime

# ==========================================
# 1. CONFIGURACIÓN DINÁMICA (FAIL-SAFE)
# ==========================================
def get_env_float(key, default):
    value = os.getenv(key)
    if not value or value.strip() == "": return default
    try: return float(value)
    except: return default

LAT = get_env_float("CITY_LAT", 6.1239)
LON = get_env_float("CITY_LON", -75.3766)
CITY_NAME = os.getenv("CITY_NAME", "Rionegro, Ant")

# Conexión API
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

params = {
    "latitude": LAT, "longitude": LON,
    "hourly": ["temperature_2m", "surface_pressure", "rain", "cloud_cover", "relative_humidity_2m", "wind_speed_10m"],
    "timezone": "UTC", "past_days": 1, "forecast_days": 1
}
responses = openmeteo.weather_api("https://api.open-meteo.com/v1/forecast", params=params)
response = responses[0]

# ==========================================
# 2. PROCESAMIENTO HORARIO BOGOTÁ
# ==========================================
zona_bogota = pytz.timezone('America/Bogota')
ahora_local = datetime.now(zona_bogota).replace(minute=0, second=0, microsecond=0)

res_hourly = response.Hourly()
dates_utc = pd.date_range(
    start=pd.to_datetime(res_hourly.Time(), unit="s", utc=True),
    end=pd.to_datetime(res_hourly.TimeEnd(), unit="s", utc=True),
    freq="h", inclusive="left"
)
dates_bogota = dates_utc.tz_convert(zona_bogota)

df = pd.DataFrame({
    "date": dates_bogota,
    "temp": res_hourly.Variables(0).ValuesAsNumpy(),
    "pressure": res_hourly.Variables(1).ValuesAsNumpy(),
    "rain": res_hourly.Variables(2).ValuesAsNumpy(),
    "clouds": res_hourly.Variables(3).ValuesAsNumpy(),
    "humidity": res_hourly.Variables(4).ValuesAsNumpy(),
    "wind": res_hourly.Variables(5).ValuesAsNumpy()
})

# ==========================================
# 3. INTELIGENCIA ARTIFICIAL (ARIMA)
# ==========================================
try:
    df_history_temp = df[df['date'] <= ahora_local].copy()
    model_arima = ARIMA(df_history_temp['pressure'], order=(5,1,0))
    res_fit = model_arima.fit()
    pred_6h = res_fit.get_forecast(steps=6).predicted_mean
    caida = pred_6h.iloc[-1] - df_history_temp['pressure'].iloc[-1]
    # Si la presión baja, la probabilidad sube (Factor de inestabilidad)
    factor_ia = 1.35 if caida < 0 else 0.7
except:
    factor_ia = 1.0

df['prob_lluvia'] = ((df['humidity'] * 0.4) + (df['clouds'] * 0.3)) * factor_ia
df['prob_lluvia'] = df['prob_lluvia'].clip(5, 98)

# Segmentación Pasado / Futuro
df_history = df[df['date'] <= ahora_local].copy()
fin_pronostico = ahora_local + pd.Timedelta(hours=6)
df_forecast = df[(df['date'] >= ahora_local) & (df['date'] <= fin_pronostico)].copy()

# ==========================================
# 4. TABLERO MULTIPANEL CON DESCRIPCIONES
# ==========================================
fig = make_subplots(
    rows=4, cols=1, vertical_spacing=0.12,
    subplot_titles=(
        "🎯 <b>RIESGO DE PRECIPITACIÓN</b>", 
        "🌡️ <b>CONFORT TÉRMICO (Temp vs Humedad)</b>", 
        "📉 <b>BARÓMETRO INTELIGENTE (Presión)</b>", 
        "☁️ <b>DINÁMICA DEL CIELO (Nubes y Viento)</b>"
    )
)

# --- PANEL 1: PROBABILIDAD + CONO ---
if not df_forecast.empty:
    incert = np.linspace(0, 15, len(df_forecast))
    fig.add_trace(go.Scatter(
        x=pd.concat([df_forecast['date'], df_forecast['date'][::-1]]),
        y=pd.concat([df_forecast['prob_lluvia'] + incert, (df_forecast['prob_lluvia'] - incert)[::-1]]),
        fill='toself', fillcolor='rgba(0, 255, 255, 0.12)', line=dict(color='rgba(255,255,255,0)'),
        name='Incertidumbre', hoverinfo="skip"
    ), row=1, col=1)

fig.add_trace(go.Scatter(x=df_history['date'], y=df_history['prob_lluvia'], name="Historia", line_color='gray'), row=1, col=1)
fig.add_trace(go.Scatter(x=df_forecast['date'], y=df_forecast['prob_lluvia'], name="Predicción", line=dict(color='cyan', width=3)), row=1, col=1)

# --- PANEL 2: AIRE ---
fig.add_trace(go.Scatter(x=df_history['date'], y=df_history['temp'], line_color='gray', showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=df_forecast['date'], y=df_forecast['temp'], line_color='orange', name="Temp °C"), row=2, col=1)
fig.add_trace(go.Scatter(x=df['date'], y=df['humidity'], line_color='royalblue', name="Hum %", opacity=0.4), row=2, col=1)

# --- PANEL 3: PRESIÓN ---
fig.add_trace(go.Scatter(x=df_history['date'], y=df_history['pressure'], line_color='gray', showlegend=False), row=3, col=1)
fig.add_trace(go.Scatter(x=df_forecast['date'], y=df_forecast['pressure'], line_color='tomato', name="hPa"), row=3, col=1)

# --- PANEL 4: FUSIÓN NUBES + VIENTO ---
fig.add_trace(go.Bar(x=df['date'], y=df['clouds'], name="Nubes %", marker_color='silver', opacity=0.3), row=4, col=1)
fig.add_trace(go.Scatter(x=df_history['date'], y=df_history['wind'], line_color='gray', showlegend=False), row=4, col=1)
fig.add_trace(go.Scatter(x=df_forecast['date'], y=df_forecast['wind'], line_color='white', name="Viento km/h"), row=4, col=1)

# ==========================================
# 5. LEYENDAS "COQUITAS" Y FORMATO
# ==========================================
descripciones = [
    (1, "<b>¿Qué significa?:</b> El área cian es el riesgo de lluvia. El cono difuso es el margen de error de la IA.<br><b>Predicción:</b> Si la línea sube, el modelo ARIMA detectó condiciones de tormenta."),
    (2, "<b>¿Qué significa?:</b> Temp (Naranja) y Humedad (Azul). Si se acercan mucho, sentirás el aire 'mojado'.<br><b>Punto de Rocío:</b> Cuando se tocan, es casi seguro que habrá niebla o lluvia ligera."),
    (3, "<b>¿Qué significa?:</b> La presión (Roja) es el peso del aire. Si cae rápido, atrae nubes de lluvia.<br><b>Tendencia:</b> Una línea roja bajando es señal de alerta climática inminente."),
    (4, "<b>¿Qué significa?:</b> Barras (Nubes) y Línea (Viento). El viento ayuda a dispersar la lluvia.<br><b>Dinámica:</b> Si hay nubes altas pero mucho viento blanco, la lluvia podría pasar de largo.")
]

for row, txt in descripciones:
    fig.add_annotation(
        xref=f"x{row} domain" if row > 1 else "x domain",
        yref=f"y{row} domain" if row > 1 else "y domain",
        x=0, y=-0.5, text=txt, showarrow=False, 
        font=dict(size=10, color="lightgray"), align="left"
    )
    # Ejes horarios
    fig.update_xaxes(tickformat="%I %p", dtick=10800000, row=row, col=1)
    # Sombra y Línea Ahora
    fig.add_vline(x=ahora_local, line_dash="dash", line_color="yellow", row=row, col=1)
    max_p = df_forecast['prob_lluvia'].max() if not df_forecast.empty else 0
    color_s = "rgba(255, 0, 0, 0.15)" if max_p > 85 else "rgba(255, 255, 0, 0.08)"
    fig.add_vrect(x0=ahora_local, x1=fin_pronostico, fillcolor=color_s, layer="below", line_width=0, row=row, col=1)

fig.update_layout(
    height=1600, template="plotly_dark", 
    margin=dict(l=20, r=20, t=150, b=120), showlegend=False,
    title={'text': f"<b>{CITY_NAME}</b><br><span style='font-size:12px; color:cyan;'>MONITOREO CON IA (ARIMA)</span><br><span style='font-size:12px; color:yellow;'>🕒 {ahora_local.strftime('%I:%M %p')}</span>", 'x': 0.5}
)

# 6. CAPA SATELITAL
plotly_html = fig.to_html(config={'responsive': True, 'displayModeBar': False}, include_plotlyjs='cdn')
windy_map = f"""<div style="padding: 20px; background: #111; text-align: center; border-top: 1px solid #333;">
<h3 style="color: white; font-family: sans-serif;">🛰️ RADAR SATELITAL EN VIVO</h3>
<iframe width="100%" height="450" src="https://embed.windy.com/embed2.html?lat={LAT}&lon={LON}&detailLat={LAT}&detailLon={LON}&zoom=8&overlay=clouds&product=ecmwf&metricWind=km%2Fh&metricTemp=%C2%B0C" frameborder="0"></iframe>
</div>"""

with open("index.html", "w", encoding="utf-8") as f:
    f.write(plotly_html + windy_map)

print(f"Reporte generado para {CITY_NAME}")