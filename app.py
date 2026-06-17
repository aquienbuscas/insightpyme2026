import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
from modelo import (calcular_prediccion, calcular_confiabilidad,
                    generar_historial_sintetico, entrenar_rf)
from clima import obtener_clima_cached
import streamlit_authenticator as stauth

# ── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="InsightPyme Agrícola",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Autenticación ─────────────────────────────────────────────────────────────
# Usuario de prototipo: juan.perez / insightpyme2024
# En la versión final esto vendrá de una base de datos SQLite
_credentials = {
    "usernames": {
        "juan.perez": {
            "name": "Juan Pérez",
            "password": "$2b$12$rEvPmYiDQ0mbMmgxZJr3tu93Or0QwTN.PxnPf2mMZj/Rt1NSqxtFS",
        }
    }
}

_authenticator = stauth.Authenticate(
    credentials=_credentials,
    cookie_name="insightpyme_cookie",
    cookie_key="insightpyme_secret_key_2026",
    cookie_expiry_days=1,
)

# Pantalla de login
_authenticator.login(location="main")

# Si no está autenticado → mostrar mensaje y detener
if not st.session_state.get("authentication_status"):
    if st.session_state.get("authentication_status") is False:
        st.error("❌ Usuario o contraseña incorrectos")
    else:
        st.markdown("""
        <div style='text-align:center; margin-top:60px;'>
            <h1 style='color:#1D7A55;'>🌱 InsightPyme</h1>
            <p style='color:#555; font-size:16px;'>Predicción hídrica con IA para agricultores de la IV Región</p>
            <br>
            <p style='color:#888; font-size:13px;'>Ingresa tus credenciales para acceder a tu predio</p>
            <p style='color:#aaa; font-size:12px; margin-top:32px;'>
                🔑 Usuario de demo: <code>juan.perez</code> &nbsp;·&nbsp; Contraseña: <code>insightpyme2026</code>
            </p>
        </div>
        """, unsafe_allow_html=True)
    st.stop()

# ── Estilos CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #f7f9f7; }

    /* Métricas */
    [data-testid="metric-container"] {
        background-color: #f0f7f4;
        border: 1px solid #d0e8dc;
        border-radius: 8px;
        padding: 12px 16px;
    }

    /* Título principal */
    .titulo-verde { color: #1D7A55; font-weight: 700; }

    /* Alerta personalizada */
    .alerta-warn {
        background-color: #fff8e1;
        border-left: 4px solid #f5a623;
        border-radius: 4px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    .alerta-ok {
        background-color: #e8f6f0;
        border-left: 4px solid #1D7A55;
        border-radius: 4px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    .alerta-danger {
        background-color: #fff0f0;
        border-left: 4px solid #e24b4a;
        border-radius: 4px;
        padding: 12px 16px;
        margin: 8px 0;
    }

    /* Badge confiabilidad */
    .badge-confiabilidad {
        display: inline-block;
        background-color: #1D7A55;
        color: white;
        border-radius: 12px;
        padding: 3px 12px;
        font-size: 13px;
        font-weight: 600;
    }

    /* Footer */
    .footer-txt {
        font-size: 12px;
        color: #888;
        text-align: center;
        margin-top: 24px;
        padding-top: 12px;
        border-top: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# ── Datos estáticos del agricultor de ejemplo ────────────────────────────────
AGRICULTOR = {
    "nombre": "Juan Pérez",
    "comuna": "Ovalle",
    "region": "IV Región de Coquimbo",
    "fuente_agua": "Canal de riego (Valle del Limarí)",
    "cultivos": [
        {"nombre": "Palto",    "hectareas": 1.5, "kc": 0.95},
        {"nombre": "Chirimoyo","hectareas": 0.8, "kc": 0.70},
        {"nombre": "Nogal",    "hectareas": 0.5, "kc": 0.90},
    ]
}

# ── Datos climáticos desde Open-Meteo (pronóstico 7 días) ───────────────────
# Coordenadas del predio de Juan Pérez (Ovalle, IV Región)
_clima_raw = obtener_clima_cached(lat=-30.60, lon=-71.20)

CLIMA_HOY = {
    "temperatura":   _clima_raw["temperatura"],
    "t_max":         _clima_raw["t_max"],
    "t_min":         _clima_raw["t_min"],
    "humedad":       _clima_raw["humedad"],
    "precipitacion": _clima_raw["precipitacion"],
    "viento":        _clima_raw["viento"],
    "descripcion":   _clima_raw["descripcion"],
    "fuente":        _clima_raw["fuente"],
    "pronostico_7d": _clima_raw["pronostico_7d"],
}

# ── Historial sintético de 52 semanas + entrenamiento del RF ─────────────────
HISTORIAL = generar_historial_sintetico(AGRICULTOR["cultivos"], semanas=52)

# Renombrar columna para compatibilidad con página historial
HISTORIAL["consumo_predicho"] = HISTORIAL["prediccion_fao"]
HISTORIAL["semana"] = [f"Sem {i}" for i in range(1, len(HISTORIAL) + 1)]

# Entrenar el Random Forest sobre el historial completo
RF_MODELO, RF_METRICAS = entrenar_rf(HISTORIAL)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌱 InsightPyme")
    st.markdown("---")

    # Info del agricultor
    st.markdown(f"**👤 {AGRICULTOR['nombre']}**")
    st.markdown(f"📍 {AGRICULTOR['comuna']}, {AGRICULTOR['region']}")
    st.markdown(f"💧 {AGRICULTOR['fuente_agua']}")
    st.markdown("---")

    # Clima
    fuente_tag = "🛰 API en vivo" if CLIMA_HOY.get('fuente') == 'api' else "📊 Datos históricos"
    st.markdown(f"**🌤 Pronóstico 7 días** · *{fuente_tag}*")
    st.markdown(f"_{CLIMA_HOY['descripcion']}_")
    col_t, col_h = st.columns(2)
    col_t.metric("Temp. media", f"{CLIMA_HOY['temperatura']}°C")
    col_h.metric("Humedad", f"{CLIMA_HOY['humedad']}%")
    col_p, col_v = st.columns(2)
    col_p.metric("Lluvia", f"{CLIMA_HOY['precipitacion']} mm")
    col_v.metric("Viento", f"{CLIMA_HOY['viento']} km/h")
    st.markdown("---")

    # Navegación
    st.markdown("**Menú**")
    pagina = st.radio(
        label="",
        options=["📊 Dashboard", "📥 Ingresar consumo", "🌾 Mi predio", "📈 Historial"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    _authenticator.logout("Cerrar sesión", location="sidebar")
    st.markdown('<p style="font-size:11px; color:#aaa;">InsightPyme · Prototipo Streamlit<br>Neg. desde la Ingeniería · Prof. Guzmán</p>', unsafe_allow_html=True)
# ── PÁGINA: DASHBOARD ─────────────────────────────────────────────────────────
if pagina == "📊 Dashboard":

    fecha_inicio = date.today()
    fecha_fin    = date.today() + timedelta(days=7)
    st.markdown(f'<h2 class="titulo-verde">Predicción semanal</h2>', unsafe_allow_html=True)
    st.markdown(f"Semana del **{fecha_inicio.strftime('%d %b')}** al **{fecha_fin.strftime('%d %b %Y')}** · {AGRICULTOR['comuna']}")
    st.markdown("---")

    # Calcular predicción con el modelo
    resultados = calcular_prediccion(AGRICULTOR["cultivos"], CLIMA_HOY, HISTORIAL, RF_MODELO, RF_METRICAS)
    total_predicho = sum(r["m3_predichos"] for r in resultados)
    promedio_historico = HISTORIAL["consumo_real"].mean()
    diferencia = total_predicho - promedio_historico
    confiabilidad = calcular_confiabilidad(HISTORIAL, RF_METRICAS)

    # ── Métricas superiores ──
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "💧 Total predicho (7 días)",
        f"{total_predicho:.0f} m³",
        delta=f"{diferencia:+.0f} m³ vs promedio",
        delta_color="inverse"
    )
    col2.metric(
        "📊 Promedio histórico",
        f"{promedio_historico:.0f} m³",
        help="Promedio de las últimas 8 semanas"
    )
    col3.metric(
        "🌡 ETo estimada",
        f"{resultados[0]['eto']:.2f} mm/día",
        help="Evapotranspiración de referencia (FAO Penman-Monteith simplificado)"
    )
    col4.metric(
        "🌾 Cultivos activos",
        f"{len(AGRICULTOR['cultivos'])}",
        help="Número de cultivos registrados en el predio"
    )

    st.markdown("---")

    # ── Desglose por cultivo ──
    st.markdown("#### Desglose por cultivo")

    for r in resultados:
        pct = r["m3_predichos"] / total_predicho
        estado = r["estado"]

        with st.container():
            col_nombre, col_ha, col_m3, col_estado, col_barra = st.columns([2, 1, 1.2, 1.3, 2.5])

            col_nombre.markdown(f"**{r['cultivo']}**")
            col_nombre.markdown(f'<span style="font-size:11px; color:#888;">Kc = {r["kc"]} · FAO</span>', unsafe_allow_html=True)

            col_ha.markdown(f"**{r['hectareas']} há**")
            col_ha.markdown('<span style="font-size:11px; color:#888;">Superficie</span>', unsafe_allow_html=True)

            col_m3.markdown(f"**{r['m3_predichos']:.0f} m³**")
            col_m3.markdown('<span style="font-size:11px; color:#888;">Predicción</span>', unsafe_allow_html=True)

            if estado == "normal":
                col_estado.markdown('<span style="background:#e8f6f0; color:#0F6E56; padding:3px 10px; border-radius:12px; font-size:12px; font-weight:600;">✓ Normal</span>', unsafe_allow_html=True)
            elif estado == "revisar":
                col_estado.markdown('<span style="background:#fff8e1; color:#854F0B; padding:3px 10px; border-radius:12px; font-size:12px; font-weight:600;">⚠ Revisar</span>', unsafe_allow_html=True)
            else:
                col_estado.markdown('<span style="background:#fff0f0; color:#A32D2D; padding:3px 10px; border-radius:12px; font-size:12px; font-weight:600;">✗ Déficit</span>', unsafe_allow_html=True)

            col_barra.markdown(f'<div style="margin-top:8px; background:#e0e0e0; border-radius:4px; height:8px;"><div style="width:{pct*100:.0f}%; background:#1D7A55; height:8px; border-radius:4px;"></div></div><span style="font-size:10px; color:#888;">{pct*100:.0f}% del total</span>', unsafe_allow_html=True)

        st.markdown("---")

    # ── Total y alertas ──
    col_total, col_alerta = st.columns([1, 2])

    with col_total:
        st.markdown("#### Resumen del predio")
        st.markdown(f"""
        | | |
        |---|---|
        | **Total predicho** | {total_predicho:.0f} m³ |
        | **Promedio histórico** | {promedio_historico:.0f} m³ |
        | **Diferencia** | {diferencia:+.0f} m³ |
        """)

    with col_alerta:
        st.markdown("#### Estado y alertas")
        alertas = [r for r in resultados if r["estado"] != "normal"]
        if not alertas:
            st.markdown('<div class="alerta-ok">✅ <strong>Todo en orden</strong><br>Tu consumo proyectado está dentro del rango normal para esta semana. No se requiere acción.</div>', unsafe_allow_html=True)
        else:
            for a in alertas:
                if a["estado"] == "revisar":
                    st.markdown(f'<div class="alerta-warn">⚠️ <strong>Atención: {a["cultivo"]}</strong><br>La reserva hídrica disponible podría ser ajustada para cubrir los {a["m3_predichos"]:.0f} m³ proyectados. Considera priorizar el riego esta semana.</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="alerta-danger">🚨 <strong>Déficit detectado: {a["cultivo"]}</strong><br>El consumo proyectado de {a["m3_predichos"]:.0f} m³ supera tu disponibilidad estimada. Acción urgente recomendada.</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Pronóstico 7 días ──
    st.markdown("#### Pronóstico climático — próximos 7 días")
    pronostico = CLIMA_HOY.get("pronostico_7d", [])
    if pronostico:
        cols_dias = st.columns(7)
        for i, dia in enumerate(pronostico):
            with cols_dias[i]:
                color_temp = "#e24b4a" if dia["t_max"] >= 30 else "#f5a623" if dia["t_max"] >= 24 else "#1D7A55"
                lluvia_ico = "🌧" if dia["precip"] > 2 else "☁️" if dia["precip"] > 0.5 else "☀️"
                html = (
                    '<div style="background:#f7f9f7;border-radius:8px;padding:8px 4px;text-align:center;border:1px solid #e0e8e4;">' +
                    f'<div style="font-size:11px;font-weight:600;color:#555;">{dia["dia"]}</div>' +
                    f'<div style="font-size:10px;color:#999;margin-bottom:4px;">{dia["fecha"]}</div>' +
                    f'<div style="font-size:18px;">{lluvia_ico}</div>' +
                    f'<div style="font-size:14px;font-weight:700;color:{color_temp};">{dia["t_max"]}°</div>' +
                    f'<div style="font-size:11px;color:#888;">{dia["t_min"]}°</div>' +
                    f'<div style="font-size:10px;color:#aaa;margin-top:2px;">{dia["precip"]} mm</div>' +
                    '</div>'
                )
                st.markdown(html, unsafe_allow_html=True)
    fuente = CLIMA_HOY.get("fuente", "fallback")
    if fuente == "api":
        st.caption("🛰 Datos en tiempo real · Open-Meteo API · Ovalle, IV Región")
    else:
        st.caption("📊 Datos históricos representativos · Climatología Ovalle · (API no disponible en este entorno)")

    st.markdown("---")

    # ── Confiabilidad del modelo ──
    st.markdown("#### Transparencia del modelo")
    col_conf, col_explicacion = st.columns([1, 2])

    with col_conf:
        color_conf = "#1D7A55" if confiabilidad >= 85 else "#f5a623" if confiabilidad >= 70 else "#e24b4a"
        st.markdown(f"""
        <div style="text-align:center; padding:20px; background:#f7f9f7; border-radius:8px; border: 1px solid #d0e8dc;">
            <div style="font-size:42px; font-weight:700; color:{color_conf};">{confiabilidad:.0f}%</div>
            <div style="font-size:13px; color:#555; margin-top:4px;">Confiabilidad del modelo</div>
            <div style="font-size:11px; color:#888; margin-top:8px;">Basado en {len(HISTORIAL)} semanas de historial</div>
        </div>
        """, unsafe_allow_html=True)

    with col_explicacion:
        # Info del RF
        peso_rf_actual = resultados[0]["peso_rf"] if resultados else 0
        st.markdown(f"""
        **¿Cómo se calcula la predicción?**

        Modelo híbrido en dos capas:

        **Capa 1 — FAO ({100 - peso_rf_actual}%)**
        > Demanda hídrica = ETo × Kc × Hectáreas

        **Capa 2 — Random Forest ({peso_rf_actual}%)**
        > Aprende el error sistemático del FAO para este agricultor.
        > Con {len(HISTORIAL)} semanas de historial, el RF corrige la predicción base.
        """)

        if RF_METRICAS and RF_METRICAS.get("mae") is not None:
            st.markdown(f"**Precisión del RF:** error promedio ±{RF_METRICAS['mae']} m³ · R² = {RF_METRICAS['r2']}")

            # Importancia de variables top 3
            if RF_METRICAS.get("importancia"):
                top3 = list(RF_METRICAS["importancia"].items())[:3]
                nombres = {"error_lag1": "Error sem. pasada", "error_lag2": "Error hace 2 sem.",
                           "consumo_lag1": "Consumo sem. pasada", "consumo_lag2": "Consumo hace 2 sem.",
                           "temperatura": "Temperatura", "t_max": "Temp. máxima",
                           "humedad": "Humedad", "sen_semana": "Estacionalidad (sen)",
                           "cos_semana": "Estacionalidad (cos)"}
                st.markdown("**Variables más influyentes:**")
                for feat, imp in top3:
                    nombre = nombres.get(feat, feat)
                    st.markdown(f"- {nombre}: `{imp*100:.1f}%`")

    st.markdown('<div class="footer-txt">InsightPyme · Predicción hídrica con IA para agricultores de la IV Región · Datos de ejemplo (prototipo)</div>', unsafe_allow_html=True)


# ── PÁGINA: INGRESAR CONSUMO ──────────────────────────────────────────────────
elif pagina == "📥 Ingresar consumo":

    st.markdown('<h2 class="titulo-verde">Ingresar consumo semanal</h2>', unsafe_allow_html=True)
    st.markdown("Reporta aquí cuánta agua usaste realmente esta semana. Este dato mejora la precisión del modelo.")
    st.markdown("---")

    semana_str = f"{(date.today() - timedelta(days=7)).strftime('%d/%m')} — {date.today().strftime('%d/%m/%Y')}"
    st.markdown(f"**Semana a reportar:** {semana_str}")

    cultivos = AGRICULTOR["cultivos"]
    total_ponderado = sum(c["hectareas"] * c["kc"] for c in cultivos)

    # ── Checkbox FUERA del form para que reaccione al instante ──
    desglosar = st.checkbox(
        "Quiero desglosar por cultivo (opcional — mejora la precisión del modelo)",
        value=False,
        key="desglosar_checkbox",
        help="Activa esto para indicar cuánta agua usó cada cultivo por separado. Si no, el modelo distribuye proporcionalmente según el Kc de cada cultivo."
    )

    st.markdown("---")

    # ── Total — también fuera del form para que el desglose se actualice en vivo ──
    consumo_total = st.number_input(
        "¿Cuántos m³ de agua usaste esta semana en total?",
        min_value=0,
        max_value=5000,
        value=510,
        step=10,
        key="consumo_total_input",
        help="Total de metros cúbicos consumidos en el predio durante los últimos 7 días"
    )

    consumos_cultivo = {}

    if desglosar:
        st.markdown("**Desglose por cultivo**")
        st.caption("Pre-rellenado con la distribución proporcional. Ajusta solo si difiere de tu riego real.")

        cols = st.columns(len(cultivos))
        suma_parciales = 0

        for i, cultivo in enumerate(cultivos):
            proporcion = (cultivo["hectareas"] * cultivo["kc"]) / total_ponderado
            valor_default = int(round(consumo_total * proporcion / 5) * 5)
            with cols[i]:
                val = st.number_input(
                    f"{cultivo['nombre']} ({cultivo['hectareas']} há)",
                    min_value=0,
                    max_value=5000,
                    value=valor_default,
                    step=5,
                    key=f"desglose_{i}",
                    help=f"Kc FAO = {cultivo['kc']}"
                )
                consumos_cultivo[cultivo["nombre"]] = val
                suma_parciales += val

        diferencia_form = abs(suma_parciales - consumo_total)
        if diferencia_form > 20:
            st.warning(f"⚠️ La suma por cultivo ({suma_parciales} m³) difiere del total ({consumo_total} m³) en {diferencia_form} m³.")
        else:
            st.success(f"✅ Los valores cuadran correctamente (diferencia: {diferencia_form} m³)")

    else:
        st.markdown("**Distribución calculada automáticamente:**")
        cols = st.columns(len(cultivos))
        for i, cultivo in enumerate(cultivos):
            proporcion = (cultivo["hectareas"] * cultivo["kc"]) / total_ponderado
            m3_estimado = round(consumo_total * proporcion)
            consumos_cultivo[cultivo["nombre"]] = m3_estimado
            with cols[i]:
                st.markdown(f"""
                <div style="background:#f0f7f4; border-radius:8px; padding:10px; text-align:center; border:1px solid #d0e8dc;">
                    <div style="font-size:11px; color:#555;">{cultivo['nombre']}</div>
                    <div style="font-size:20px; font-weight:700; color:#1D7A55;">{m3_estimado} m³</div>
                    <div style="font-size:10px; color:#888;">{proporcion*100:.0f}% del total</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Form solo para las notas y el botón de guardar ──
    with st.form("form_guardar"):
        notas = st.text_area(
            "Notas opcionales (ej: hubo corte de canal, riego de emergencia, etc.)",
            placeholder="Escribe aquí si ocurrió algo inusual esta semana...",
            height=68
        )
        submitted = st.form_submit_button("✅ Guardar consumo semanal", use_container_width=True)

        if submitted:
            if desglosar:
                detalle = " | ".join([f"{k}: {v} m³" for k, v in consumos_cultivo.items()])
                st.success(f"✅ Consumo de **{consumo_total} m³** registrado con desglose: {detalle}")
            else:
                st.success(f"✅ Consumo de **{consumo_total} m³** registrado. Distribución proporcional aplicada.")
            st.info("El modelo actualizará sus predicciones con este nuevo dato.")

    st.markdown("---")
    st.markdown("#### Último consumo registrado")
    ultima = HISTORIAL.iloc[-1]
    col1, col2, col3 = st.columns(3)
    col1.metric("Semana anterior", f"{ultima['consumo_real']} m³", help="Consumo real reportado")
    col2.metric("Predicción de esa semana", f"{ultima['consumo_predicho']} m³")
    col3.metric("Error del modelo", f"{abs(ultima['consumo_real'] - ultima['consumo_predicho'])} m³",
                delta=f"{abs(ultima['consumo_real']-ultima['consumo_predicho'])/ultima['consumo_real']*100:.1f}%",
                delta_color="inverse")


# ── PÁGINA: MI PREDIO ─────────────────────────────────────────────────────────
elif pagina == "🌾 Mi predio":

    st.markdown('<h2 class="titulo-verde">Mi predio</h2>', unsafe_allow_html=True)
    st.markdown("Datos registrados de tu predio. En la versión final podrás editar estos datos.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Información general")
        st.markdown(f"""
        | Campo | Valor |
        |---|---|
        | **Nombre** | {AGRICULTOR['nombre']} |
        | **Comuna** | {AGRICULTOR['comuna']} |
        | **Región** | {AGRICULTOR['region']} |
        | **Fuente de agua** | {AGRICULTOR['fuente_agua']} |
        """)

    with col2:
        st.markdown("#### Cultivos registrados")
        df_cultivos = pd.DataFrame(AGRICULTOR["cultivos"])
        df_cultivos.columns = ["Cultivo", "Hectáreas", "Coeficiente Kc (FAO)"]
        st.dataframe(df_cultivos, use_container_width=True, hide_index=True)
        total_ha = sum(c["hectareas"] for c in AGRICULTOR["cultivos"])
        st.markdown(f"**Total del predio:** {total_ha} hectáreas")

    st.markdown("---")
    st.markdown("#### Coeficientes Kc — ¿qué significan?")
    st.markdown("""
    El coeficiente **Kc** (coeficiente de cultivo) es un valor estándar definido por la **FAO** que indica
    cuánta agua necesita cada tipo de cultivo en relación a la evapotranspiración de referencia (ETo).

    | Cultivo | Kc (etapa media) | Demanda relativa |
    |---|---|---|
    | Palto | 0.95 | Alta |
    | Chirimoyo | 0.70 | Media |
    | Nogal | 0.90 | Alta |
    | Vid | 0.70 | Media |
    | Olivo | 0.65 | Media-baja |
    | Maíz | 1.20 | Muy alta |
    | Trigo | 1.10 | Alta |
    *Fuente: FAO Irrigation and Drainage Paper 56 (Allen et al., 1998)*
    """)


# ── PÁGINA: HISTORIAL ─────────────────────────────────────────────────────────
elif pagina == "📈 Historial":

    st.markdown('<h2 class="titulo-verde">Historial de consumo</h2>', unsafe_allow_html=True)
    st.markdown("Comparación entre el consumo real reportado y las predicciones del modelo.")
    st.markdown("---")

    # Tabla — seleccionar solo columnas relevantes
    df_display = HISTORIAL[["semana", "fecha", "consumo_real", "prediccion_fao", "error_fao"]].copy()
    df_display["error_pct"] = (df_display["error_fao"].abs() / df_display["consumo_real"] * 100).round(1).astype(str) + "%"
    df_display["error_fao"] = df_display["error_fao"].abs()
    df_display.columns = ["Semana", "Fecha", "Consumo real (m³)", "Predicción FAO (m³)", "Error abs. (m³)", "Error (%)"]

    # Mostrar últimas 12 semanas por defecto
    mostrar_todo = st.checkbox("Mostrar las 52 semanas", value=False)
    df_mostrar = df_display if mostrar_todo else df_display.tail(12)
    st.dataframe(df_mostrar, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Gráfico consumo real vs predicción FAO
    st.markdown("#### Consumo real vs. predicción FAO (últimas 26 semanas)")
    df_chart = HISTORIAL.tail(26)[["semana", "consumo_real", "prediccion_fao"]].set_index("semana")
    df_chart.columns = ["Consumo real (m³)", "Predicción FAO (m³)"]
    st.line_chart(df_chart, color=["#1D7A55", "#f5a623"])

    st.markdown("---")

    # Métricas de precisión del modelo híbrido
    st.markdown("#### Métricas del modelo híbrido")

    # Error del FAO puro
    mae_fao  = HISTORIAL["error_fao"].abs().mean()
    mape_fao = (HISTORIAL["error_fao"].abs() / HISTORIAL["consumo_real"] * 100).mean()

    # Error del RF (sobre los datos de entrenamiento)
    mae_rf  = RF_METRICAS.get("mae", mae_fao) if RF_METRICAS else mae_fao
    r2_rf   = RF_METRICAS.get("r2", 0) if RF_METRICAS else 0
    # Calcular peso_rf directamente desde el historial (no depende del dashboard)
    n_sem = len(HISTORIAL)
    if n_sem < 8:
        peso_rf = 0
    elif n_sem < 16:
        peso_rf = 30
    elif n_sem < 26:
        peso_rf = 50
    else:
        peso_rf = 75

    # Confiabilidad basada en error del modelo híbrido
    # Estimamos el error híbrido como reducción proporcional al peso del RF
    mejora_rf = (1 - mae_rf / mae_fao) * peso_rf / 100 if mae_fao > 0 else 0
    mae_hibrido = mae_fao * (1 - mejora_rf)
    mape_hibrido = mape_fao * (1 - mejora_rf)
    confiabilidad = calcular_confiabilidad(HISTORIAL, RF_METRICAS)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE modelo FAO", f"{mae_fao:.1f} m³",
                help="Error promedio del modelo físico puro")
    col2.metric("MAE modelo RF", f"{mae_rf:.1f} m³",
                help="Error promedio del Random Forest (cross-validation)")
    col3.metric("R² del Random Forest", f"{r2_rf:.3f}",
                help="Qué tan bien explica el RF el error del FAO (1.0 = perfecto)")
    col4.metric("Confiabilidad híbrida", f"{confiabilidad:.0f}%",
                help="Precisión global del modelo combinado FAO + RF")

    st.markdown("---")

    # Importancia de variables
    if RF_METRICAS and RF_METRICAS.get("importancia"):
        st.markdown("#### Variables más influyentes en el Random Forest")
        nombres = {
            "error_lag1":   "Error FAO semana pasada",
            "error_lag2":   "Error FAO hace 2 semanas",
            "consumo_lag1": "Consumo real semana pasada",
            "consumo_lag2": "Consumo real hace 2 semanas",
            "temperatura":  "Temperatura media",
            "t_max":        "Temperatura máxima",
            "humedad":      "Humedad relativa",
            "sen_semana":   "Estacionalidad (seno)",
            "cos_semana":   "Estacionalidad (coseno)",
        }
        imp_data = {nombres.get(k, k): round(v * 100, 1)
                    for k, v in RF_METRICAS["importancia"].items()}
        df_imp = pd.DataFrame(list(imp_data.items()),
                              columns=["Variable", "Importancia (%)"])
        st.bar_chart(df_imp.set_index("Variable"), color="#1D7A55")

        st.caption("La importancia indica cuánto contribuye cada variable a la corrección del Random Forest.")

    st.markdown("""
    > **¿Por qué el modelo mejora con el tiempo?**
    > El Random Forest aprende el error sistemático que comete el modelo FAO para este agricultor
    > específico. Con más semanas de historial, el RF tiene más ejemplos y su corrección
    > se vuelve más precisa — reduciendo el error final del sistema híbrido.
    """)
