# InsightPyme — Prototipo v1.0

Plataforma de predicción hídrica con IA para agricultores de la IV Región de Coquimbo.

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

```
insightpyme/
├── app.py          # Interfaz Streamlit (frontend)
├── modelo.py       # Motor de predicción FAO ETo × Kc (backend)
├── requirements.txt
└── README.md
```

## Modelo de predicción

El modelo usa el estándar agronómico de la FAO:

```
Demanda hídrica = ETo × Kc × Hectáreas × 7 días × 10
```

- **ETo**: Estimada con la ecuación de Hargreaves-Samani
- **Kc**: Coeficiente de cultivo estándar FAO (Paper 56)
- **Ajuste con historial**: la predicción se pondera con el consumo real histórico del agricultor

## Páginas disponibles

| Página | Descripción |
|---|---|
| Dashboard | Predicción semanal con desglose por cultivo y alertas |
| Ingresar consumo | Formulario de reporte semanal |
| Mi predio | Datos del agricultor y cultivos registrados |
| Historial | Gráfico y métricas de precisión del modelo |

## Notas del prototipo v1.0

- Datos estáticos del agricultor (Juan Pérez, Ovalle)
- Clima representativo de Ovalle en temporada
- Historial de 8 semanas de ejemplo realista
- Sin login/autenticación (pendiente v2.0)
- Sin conexión a API climática real (pendiente v2.0)

## Próximas versiones

- **v1.5**: Subida de historial desde Excel
- **v2.0**: Login con streamlit-authenticator + API climática real (DMC o NASA POWER)
- **v2.5**: Múltiples agricultores + modelo compartido que mejora con más datos
