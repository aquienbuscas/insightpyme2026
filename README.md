# InsightPyme — Prototipo v1.0
Plataforma de predicción hídrica con IA para agricultores de la IV Región de Coquimbo.

## Credenciales de acceso
- **Usuario:** `juan.perez`
- **Contraseña:** `insightpyme2026`

## Cómo correr el prototipo

### 1. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 2. Ejecutar
```bash
streamlit run app.py
```

Se abrirá automáticamente en `http://localhost:8501`

---

## Estructura del proyecto

nsightpyme/

├── app.py            --> Interfaz Streamlit (frontend + autenticación)

├── modelo.py         --> Motor de predicción FAO ETo × Kc + Random Forest

├── clima.py          --> Integración API climática Open-Meteo (pronóstico 7 días)

├── requirements.txt

└── README.md

---

## Funcionalidades

| Página | Descripción |
|---|---|
| Login | Autenticación con usuario y contraseña |
| Dashboard | Predicción semanal con desglose por cultivo, pronóstico 7 días y alertas |
| Ingresar consumo | Reporte semanal de consumo real con desglose opcional por cultivo |
| Mi predio | Datos del agricultor, cultivos registrados y tabla de coeficientes Kc FAO |
| Historial | Gráfico consumo real vs. predicción, métricas del modelo e importancia de variables |

---

## Modelo de predicción

Arquitectura híbrida en dos capas:

**Capa 1 — Modelo físico FAO**

Demanda hídrica = ETo × Kc × Hectáreas × 7 días × 10

- **ETo**: Estimada con Hargreaves-Samani usando pronóstico real de temperatura, humedad y viento
- **Kc**: Coeficiente de cultivo estándar FAO Paper 56

**Capa 2 — Random Forest (residual learning)**
- Aprende el error sistemático que comete el modelo FAO para este agricultor específico
- Se entrena sobre 52 semanas de historial sintético con patrones estacionales reales
- La ponderación FAO/RF es dinámica según el historial disponible

---

## Integración climática

- **API:** Open-Meteo (gratuita, sin registro)
- **Datos:** pronóstico real de los próximos 7 días — temperatura máxima/mínima, humedad, viento y precipitación
- **Fallback:** si la API no está disponible, usa climatología histórica representativa de Ovalle
