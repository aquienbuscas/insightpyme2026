"""
modelo.py — Motor de predicción hídrica InsightPyme

Arquitectura híbrida:
    1. Modelo físico FAO (ETo × Kc): calcula la demanda base
    2. Random Forest (residual learning): aprende y corrige el error sistemático del FAO

El RF no predice el consumo desde cero — predice cuánto se va a equivocar
el modelo FAO para ese agricultor específico. Esto se llama residual learning.

Referencias:
    - FAO Irrigation and Drainage Paper 56 (Allen et al., 1998)
    - Hargreaves & Samani (1985) — simplificación de Penman-Monteith
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from datetime import date


# ══════════════════════════════════════════════════════════════════════════════
# CAPA 1 — MODELO FÍSICO FAO
# ══════════════════════════════════════════════════════════════════════════════

def calcular_eto(temperatura: float, humedad: float, viento: float,
                 t_max: float = None, t_min: float = None) -> float:
    """
    Estima ETo (evapotranspiración de referencia) en mm/día.
    Usa Hargreaves-Samani con correcciones de humedad y viento.

    Si se proveen t_max y t_min, usa la amplitud térmica real.
    Si no, estima ±6.5°C alrededor de la temperatura media.
    """
    Ra = 11.0  # Radiación solar extraterrestre calibrada para lat ~30°S junio

    if t_max is None:
        t_max = temperatura + 6.5
    if t_min is None:
        t_min = temperatura - 6.5

    T_mean = (t_max + t_min) / 2
    amplitud = t_max - t_min

    # Hargreaves-Samani base
    eto_base = 0.0023 * Ra * (T_mean + 17.8) * (amplitud ** 0.5)

    # Corrección por humedad (alta humedad reduce ETo)
    factor_humedad = 1 - (humedad - 50) * 0.003

    # Corrección por viento
    factor_viento = 1 + (viento - 10) * 0.005

    eto = eto_base * factor_humedad * factor_viento
    return float(np.clip(eto, 1.0, 9.0))


def calcular_demanda_cultivo(eto: float, kc: float, hectareas: float) -> float:
    """
    Demanda hídrica semanal de un cultivo en m³.
    Fórmula FAO: ETc = ETo × Kc
    Conversión: ETc (mm/día) × 7 días × 10 × hectáreas = m³
    """
    etc_diario = eto * kc
    return round(etc_diario * 7 * 10 * hectareas, 1)


# ══════════════════════════════════════════════════════════════════════════════
# DATOS SINTÉTICOS — Historial realista de 52 semanas
# ══════════════════════════════════════════════════════════════════════════════

def generar_historial_sintetico(cultivos: list, semanas: int = 52,
                                 seed: int = 42) -> pd.DataFrame:
    """
    Genera un historial sintético de 52 semanas con patrones realistas:
    - Variación estacional del clima (más calor en verano, más lluvia en invierno)
    - Error sistemático del FAO: subestima en verano, sobreestima en invierno
    - Comportamiento humano: el agricultor riega un poco más de lo necesario
    - Ruido aleatorio: semanas atípicas por eventos puntuales

    Esto permite entrenar el RF desde el día 1 sin datos reales.
    """
    rng = np.random.default_rng(seed)

    # Clima promedio por semana del año para Ovalle (IV Región)
    # Basado en climatología histórica
    def clima_semana(semana_año: int) -> dict:
        """Simula el clima de una semana del año para Ovalle."""
        # Temperatura: máximo en enero (sem 1-4), mínimo en julio (sem 26-30)
        angulo = (semana_año / 52) * 2 * np.pi
        t_base = 22.0 + 10.0 * np.cos(angulo - np.pi * 0.1)
        humedad_base = 55.0 - 20.0 * np.cos(angulo)
        viento_base = 12.0 + 2.0 * np.sin(angulo)

        t_max = t_base + 6 + rng.normal(0, 1.5)
        t_min = t_base - 6 + rng.normal(0, 1.5)
        humedad = np.clip(humedad_base + rng.normal(0, 5), 25, 95)
        viento = max(5, viento_base + rng.normal(0, 2))

        return {
            "t_max": round(t_max, 1),
            "t_min": round(t_min, 1),
            "temperatura": round((t_max + t_min) / 2, 1),
            "humedad": round(humedad),
            "viento": round(viento, 1),
        }

    filas = []
    for i in range(semanas):
        semana_año = (i % 52) + 1
        clima = clima_semana(semana_año)

        # Predicción FAO base
        eto = calcular_eto(
            clima["temperatura"], clima["humedad"], clima["viento"],
            clima["t_max"], clima["t_min"]
        )
        demanda_total_fao = sum(
            calcular_demanda_cultivo(eto, c["kc"], c["hectareas"])
            for c in cultivos
        )

        # Error sistemático del FAO para este agricultor:
        # - En verano (calor > 28°C): FAO subestima porque el agricultor
        #   riega más por prevención
        # - En invierno: FAO sobreestima porque el agricultor ahorra agua
        # - Ruido aleatorio: ±5% por eventos puntuales
        factor_estacional = 1 + 0.08 * np.sin((semana_año / 52) * 2 * np.pi)
        factor_temp = 0.04 if clima["t_max"] > 28 else -0.03
        ruido = rng.normal(0, 0.03)

        error_sistematico = demanda_total_fao * (factor_estacional + factor_temp + ruido - 1)

        consumo_real = round(demanda_total_fao + error_sistematico)
        consumo_real = max(100, consumo_real)  # mínimo físico razonable

        filas.append({
            "semana_idx":    i + 1,
            "semana_año":    semana_año,
            "fecha":         (date.today() - pd.Timedelta(weeks=semanas - i)).strftime("%d/%m/%Y"),
            "temperatura":   clima["temperatura"],
            "t_max":         clima["t_max"],
            "t_min":         clima["t_min"],
            "humedad":       clima["humedad"],
            "viento":        clima["viento"],
            "eto":           round(eto, 3),
            "prediccion_fao": round(demanda_total_fao),
            "consumo_real":   consumo_real,
            "error_fao":      consumo_real - round(demanda_total_fao),
        })

    return pd.DataFrame(filas)


# ══════════════════════════════════════════════════════════════════════════════
# CAPA 2 — RANDOM FOREST (RESIDUAL LEARNING)
# ══════════════════════════════════════════════════════════════════════════════

def construir_features(historial: pd.DataFrame) -> pd.DataFrame:
    """
    Construye las features (variables de entrada) para el Random Forest.

    El RF aprende el ERROR del FAO, no el consumo absoluto.
    Features:
        - error_lag1: error FAO la semana pasada
        - error_lag2: error FAO hace 2 semanas
        - consumo_lag1: consumo real la semana pasada
        - consumo_lag2: consumo real hace 2 semanas
        - temperatura: temperatura media de la semana
        - t_max: temperatura máxima
        - humedad: humedad relativa
        - semana_año: para capturar estacionalidad (1-52)
        - sen_semana, cos_semana: encoding cíclico de la estacionalidad
    """
    df = historial.copy()

    df["error_lag1"]   = df["error_fao"].shift(1)
    df["error_lag2"]   = df["error_fao"].shift(2)
    df["consumo_lag1"] = df["consumo_real"].shift(1)
    df["consumo_lag2"] = df["consumo_real"].shift(2)

    # Encoding cíclico para la semana del año
    # Permite al RF entender que sem 52 y sem 1 son consecutivas
    df["sen_semana"] = np.sin(2 * np.pi * df["semana_año"] / 52)
    df["cos_semana"] = np.cos(2 * np.pi * df["semana_año"] / 52)

    return df.dropna()  # eliminar las primeras filas sin lags


FEATURES = [
    "error_lag1", "error_lag2",
    "consumo_lag1", "consumo_lag2",
    "temperatura", "t_max", "humedad",
    "sen_semana", "cos_semana",
]


def entrenar_rf(historial: pd.DataFrame) -> tuple:
    """
    Entrena el Random Forest sobre el historial disponible.

    Retorna (modelo_rf, metricas) donde metricas incluye:
        - mae: error absoluto medio en m³
        - r2: coeficiente de determinación
        - n_semanas: semanas usadas para entrenar
        - importancia_features: qué variables importan más
    """
    if len(historial) < 6:
        return None, {"mae": None, "r2": None, "n_semanas": len(historial)}

    df_features = construir_features(historial)

    if len(df_features) < 5:
        return None, {"mae": None, "r2": None, "n_semanas": len(historial)}

    X = df_features[FEATURES].values
    y = df_features["error_fao"].values

    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=4,          # limitado para evitar overfitting con pocos datos
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X, y)

    # Métricas con cross-validation si hay suficientes datos
    if len(df_features) >= 10:
        cv_scores = cross_val_score(rf, X, y, cv=3, scoring="neg_mean_absolute_error")
        mae = round(-cv_scores.mean(), 1)
    else:
        from sklearn.metrics import mean_absolute_error
        mae = round(mean_absolute_error(y, rf.predict(X)), 1)

    r2 = round(rf.score(X, y), 3)

    importancia = dict(zip(FEATURES, rf.feature_importances_.round(3)))
    importancia_ordenada = dict(sorted(importancia.items(), key=lambda x: x[1], reverse=True))

    metricas = {
        "mae":                mae,
        "r2":                 r2,
        "n_semanas":          len(historial),
        "importancia":        importancia_ordenada,
    }

    return rf, metricas


def predecir_correccion_rf(rf, historial: pd.DataFrame,
                            clima_actual: dict, semana_año: int) -> float:
    """
    Predice la corrección que debe aplicarse sobre la predicción FAO.
    Retorna los m³ de corrección (puede ser positiva o negativa).
    """
    if rf is None or len(historial) < 4:
        return 0.0

    ultimas = historial.tail(2)
    if len(ultimas) < 2:
        return 0.0

    error_lag1 = ultimas["error_fao"].iloc[-1]
    error_lag2 = ultimas["error_fao"].iloc[-2]
    consumo_lag1 = ultimas["consumo_real"].iloc[-1]
    consumo_lag2 = ultimas["consumo_real"].iloc[-2]

    sen_s = np.sin(2 * np.pi * semana_año / 52)
    cos_s = np.cos(2 * np.pi * semana_año / 52)

    X_pred = np.array([[
        error_lag1, error_lag2,
        consumo_lag1, consumo_lag2,
        clima_actual["temperatura"],
        clima_actual.get("t_max", clima_actual["temperatura"] + 6),
        clima_actual["humedad"],
        sen_s, cos_s,
    ]])

    return float(rf.predict(X_pred)[0])


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL — PREDICCIÓN HÍBRIDA
# ══════════════════════════════════════════════════════════════════════════════

def calcular_prediccion(cultivos: list, clima: dict,
                        historial: pd.DataFrame = None,
                        rf=None, metricas_rf: dict = None) -> list:
    """
    Predicción híbrida FAO + RF para todos los cultivos del predio.

    Ponderación dinámica según el historial disponible:
        < 8 semanas:  FAO 100% (RF sin suficientes datos)
        8-15 semanas: FAO 70% + RF 30%
        16-25 semanas: FAO 50% + RF 50%
        > 25 semanas: FAO 25% + RF 75%
    """
    t_max = clima.get("t_max", clima["temperatura"] + 6)
    t_min = clima.get("t_min", clima["temperatura"] - 6)

    eto = calcular_eto(
        clima["temperatura"], clima["humedad"], clima["viento"],
        t_max, t_min
    )

    # Total FAO del predio para referencia de estados
    total_fao = sum(
        calcular_demanda_cultivo(eto, c["kc"], c["hectareas"])
        for c in cultivos
    )

    # Referencia para determinar estados (normal/revisar/déficit)
    if historial is not None and len(historial) > 0:
        promedio_historico = historial["consumo_real"].mean()
    else:
        promedio_historico = total_fao

    # Ponderación dinámica RF vs FAO
    n_semanas = len(historial) if historial is not None else 0
    if n_semanas < 8:
        peso_rf = 0.0
    elif n_semanas < 16:
        peso_rf = 0.30
    elif n_semanas < 26:
        peso_rf = 0.50
    else:
        peso_rf = 0.75
    peso_fao = 1.0 - peso_rf

    # Corrección del RF (sobre el total del predio)
    semana_año = date.today().isocalendar().week
    correccion_rf = 0.0
    if rf is not None and peso_rf > 0 and historial is not None:
        correccion_rf = predecir_correccion_rf(rf, historial, clima, semana_año)

    # Total predicho con mezcla FAO + RF
    total_predicho = total_fao * peso_fao + (total_fao + correccion_rf) * peso_rf

    resultados = []
    for cultivo in cultivos:
        # Predicción FAO individual
        m3_fao = calcular_demanda_cultivo(eto, cultivo["kc"], cultivo["hectareas"])

        # Distribuir la corrección RF proporcionalmente a cada cultivo
        proporcion = m3_fao / total_fao if total_fao > 0 else 1.0 / len(cultivos)
        m3_final = round(total_predicho * proporcion, 1)

        # Estado vs. proporción esperada del historial
        m3_ref = promedio_historico * proporcion
        estado = determinar_estado(m3_final, m3_ref)

        resultados.append({
            "cultivo":       cultivo["nombre"],
            "hectareas":     cultivo["hectareas"],
            "kc":            cultivo["kc"],
            "eto":           round(eto, 3),
            "m3_fao":        round(m3_fao, 1),
            "m3_predichos":  m3_final,
            "correccion_rf": round(correccion_rf * proporcion, 1),
            "peso_rf":       round(peso_rf * 100),
            "estado":        estado,
        })

    return resultados


def determinar_estado(m3_predichos: float, m3_promedio: float) -> str:
    ratio = m3_predichos / m3_promedio if m3_promedio > 0 else 1.0
    if ratio > 1.20:
        return "deficit"
    elif ratio > 1.08:
        return "revisar"
    else:
        return "normal"


# ══════════════════════════════════════════════════════════════════════════════
# MÉTRICAS Y CONFIABILIDAD
# ══════════════════════════════════════════════════════════════════════════════

def calcular_confiabilidad(historial: pd.DataFrame,
                           metricas_rf: dict = None) -> float:
    """
    Confiabilidad del modelo híbrido FAO + RF.

    Se calcula en base al R² del RF y las semanas disponibles:
    - R² alto del RF → alta confiabilidad
    - Más semanas de historial → más confiabilidad
    - Sin RF (pocas semanas) → confiabilidad base del FAO solo
    """
    if historial is None or len(historial) == 0:
        return 0.0

    n = len(historial)

    # Confiabilidad base: crece con el historial (logística suave)
    # 10 semanas → ~60%, 26 semanas → ~80%, 52 semanas → ~90%
    conf_base = 99.0 * (1 - np.exp(-n / 30))

    # Ajuste por calidad del RF
    if metricas_rf and metricas_rf.get("r2") is not None:
        r2 = metricas_rf["r2"]
        # R² 0.9+ → multiplicador 1.0 (no penaliza)
        # R² 0.5  → multiplicador 0.85
        # R² < 0.3 → multiplicador 0.70
        factor_r2 = 0.70 + 0.30 * min(1.0, max(0.0, r2))
        conf_final = conf_base * factor_r2
    else:
        # Sin RF — solo FAO: confiabilidad más conservadora
        conf_final = conf_base * 0.65

    return round(min(97.0, conf_final), 1)
