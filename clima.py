"""
clima.py — Módulo de datos climáticos para InsightPyme

Obtiene el pronóstico de los próximos 7 días desde Open-Meteo (gratuito, sin API key).
Si la API no está disponible, usa datos de respaldo representativos para Ovalle, IV Región.

Referencia API: https://open-meteo.com/en/docs
Coordenadas Ovalle: lat=-30.60, lon=-71.20
"""

import requests
from datetime import date, timedelta
import streamlit as st


# ── Coordenadas por defecto (Ovalle, IV Región) ───────────────────────────────
LAT_DEFAULT = -30.60
LON_DEFAULT = -71.20

# ── Datos de respaldo por mes para Ovalle ─────────────────────────────────────
# Basados en climatología histórica del valle del Limarí
# Fuente: registros DMC y registros agronómicos INIA Intihuasi
CLIMA_FALLBACK_POR_MES = {
    1:  {"t_max": 30.5, "t_min": 14.0, "humedad": 45, "viento": 14, "precip": 0.2, "desc": "Soleado"},
    2:  {"t_max": 30.0, "t_min": 14.2, "humedad": 47, "viento": 13, "precip": 0.1, "desc": "Soleado"},
    3:  {"t_max": 28.0, "t_min": 12.5, "humedad": 50, "viento": 12, "precip": 0.5, "desc": "Mayormente despejado"},
    4:  {"t_max": 24.0, "t_min": 10.0, "humedad": 55, "viento": 11, "precip": 3.0, "desc": "Parcialmente nublado"},
    5:  {"t_max": 20.0, "t_min": 7.5,  "humedad": 62, "viento": 10, "precip": 8.0, "desc": "Nublado con lluvias ocasionales"},
    6:  {"t_max": 17.5, "t_min": 5.5,  "humedad": 68, "viento": 10, "precip": 12.0,"desc": "Nublado"},
    7:  {"t_max": 16.5, "t_min": 4.5,  "humedad": 70, "viento": 11, "precip": 15.0,"desc": "Nublado con lluvias"},
    8:  {"t_max": 18.0, "t_min": 5.5,  "humedad": 65, "viento": 12, "precip": 8.0, "desc": "Parcialmente nublado"},
    9:  {"t_max": 21.0, "t_min": 8.0,  "humedad": 58, "viento": 13, "precip": 4.0, "desc": "Variable"},
    10: {"t_max": 24.5, "t_min": 10.5, "humedad": 52, "viento": 13, "precip": 1.5, "desc": "Mayormente despejado"},
    11: {"t_max": 27.5, "t_min": 12.5, "humedad": 47, "viento": 14, "precip": 0.5, "desc": "Soleado"},
    12: {"t_max": 30.0, "t_min": 13.5, "humedad": 44, "viento": 14, "precip": 0.2, "desc": "Soleado"},
}


def _clima_fallback(mes: int) -> dict:
    """Retorna datos climáticos de respaldo para el mes dado."""
    datos = CLIMA_FALLBACK_POR_MES[mes]
    t_mean = (datos["t_max"] + datos["t_min"]) / 2
    return {
        "temperatura":   round(t_mean, 1),
        "t_max":         datos["t_max"],
        "t_min":         datos["t_min"],
        "humedad":       datos["humedad"],
        "precipitacion": datos["precip"],
        "viento":        datos["viento"],
        "descripcion":   datos["desc"],
        "fuente":        "fallback",
        "pronostico_7d": _generar_pronostico_sintetico(datos),
    }


def _generar_pronostico_sintetico(datos: dict) -> list:
    """
    Genera un pronóstico sintético de 7 días con variación realista
    alrededor de los valores base del mes.
    """
    import random
    random.seed(42)  # reproducible para que no cambie cada vez que recarga

    pronostico = []
    for i in range(7):
        dia = date.today() + timedelta(days=i)
        # Variación diaria realista: ±2°C temperatura, ±8% humedad
        t_max = round(datos["t_max"] + random.uniform(-2.0, 2.0), 1)
        t_min = round(datos["t_min"] + random.uniform(-1.5, 1.5), 1)
        hum   = round(datos["humedad"] + random.uniform(-8, 8))
        viento= round(datos["viento"]  + random.uniform(-2, 3), 1)
        precip= round(max(0, datos["precip"] / 7 + random.uniform(-0.5, 0.5)), 1)

        pronostico.append({
            "fecha":    dia.strftime("%d/%m"),
            "dia":      ["Lun","Mar","Mié","Jue","Vie","Sáb","Dom"][dia.weekday()],
            "t_max":    t_max,
            "t_min":    t_min,
            "t_mean":   round((t_max + t_min) / 2, 1),
            "humedad":  max(20, min(100, hum)),
            "viento":   max(2, viento),
            "precip":   precip,
        })
    return pronostico


def obtener_clima(lat: float = LAT_DEFAULT, lon: float = LON_DEFAULT) -> dict:
    """
    Obtiene el pronóstico climático de los próximos 7 días desde Open-Meteo.
    Si la API falla, retorna datos de respaldo representativos para la IV Región.

    Parámetros:
        lat: Latitud del predio
        lon: Longitud del predio

    Retorna dict con:
        - temperatura: promedio semanal (°C)
        - t_max, t_min: extremos semanales
        - humedad: promedio semanal (%)
        - precipitacion: acumulado semanal (mm)
        - viento: promedio semanal (km/h)
        - descripcion: texto descriptivo
        - fuente: "api" o "fallback"
        - pronostico_7d: lista con datos de cada día
    """
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_max,temperature_2m_min,"
        f"relative_humidity_2m_mean,wind_speed_10m_max,precipitation_sum"
        f"&forecast_days=7"
        f"&timezone=America%2FSantiago"
    )

    try:
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        data = resp.json()["daily"]

        # Construir pronóstico día a día
        pronostico = []
        for i in range(7):
            fecha = date.today() + timedelta(days=i)
            t_max  = data["temperature_2m_max"][i]
            t_min  = data["temperature_2m_min"][i]
            t_mean = round((t_max + t_min) / 2, 1)
            hum    = data["relative_humidity_2m_mean"][i]
            viento = data["wind_speed_10m_max"][i]
            precip = data["precipitation_sum"][i] or 0.0

            pronostico.append({
                "fecha":   fecha.strftime("%d/%m"),
                "dia":     ["Lun","Mar","Mié","Jue","Vie","Sáb","Dom"][fecha.weekday()],
                "t_max":   round(t_max, 1),
                "t_min":   round(t_min, 1),
                "t_mean":  t_mean,
                "humedad": round(hum),
                "viento":  round(viento, 1),
                "precip":  round(precip, 1),
            })

        # Promedios semanales para el modelo
        t_max_prom  = round(sum(d["t_max"]   for d in pronostico) / 7, 1)
        t_min_prom  = round(sum(d["t_min"]   for d in pronostico) / 7, 1)
        t_mean_prom = round(sum(d["t_mean"]  for d in pronostico) / 7, 1)
        hum_prom    = round(sum(d["humedad"] for d in pronostico) / 7)
        viento_prom = round(sum(d["viento"]  for d in pronostico) / 7, 1)
        precip_total= round(sum(d["precip"]  for d in pronostico), 1)

        # Descripción automática
        if precip_total > 10:
            desc = "Semana lluviosa"
        elif precip_total > 2:
            desc = "Lluvias ocasionales"
        elif t_max_prom > 28:
            desc = "Semana cálida y seca"
        elif hum_prom > 70:
            desc = "Nublado y húmedo"
        else:
            desc = "Parcialmente nublado"

        return {
            "temperatura":   t_mean_prom,
            "t_max":         t_max_prom,
            "t_min":         t_min_prom,
            "humedad":       hum_prom,
            "precipitacion": precip_total,
            "viento":        viento_prom,
            "descripcion":   desc,
            "fuente":        "api",
            "pronostico_7d": pronostico,
        }

    except Exception:
        # Si la API falla por cualquier razón → datos de respaldo
        mes = date.today().month
        return _clima_fallback(mes)


@st.cache_data(ttl=3600)  # cachear 1 hora — no consultar la API en cada recarga
def obtener_clima_cached(lat: float = LAT_DEFAULT, lon: float = LON_DEFAULT) -> dict:
    """Versión cacheada para Streamlit — evita llamar la API en cada interacción."""
    return obtener_clima(lat, lon)
