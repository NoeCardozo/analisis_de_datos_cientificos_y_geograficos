# Análisis EEG Simplificado

## Descripción
Aplicación simplificada para comparar señales EEG del baseline con condiciones específicas (mirar con y sin anteojos).

## Estructura del Proyecto
```
python-project/
├── main.py                    # Aplicación principal simplificada
├── requirements.txt           # Dependencias mínimas
├── README.md                 # Este archivo
├── data/
│   └── dataNoelia/
│       └── carodata/
│           ├── carobaseline.dat
│           ├── caromirarsinanteojos.dat
│           └── caromirarconanteojos.dat
├── resultados/               # Archivos generados
└── venv/                    # Entorno virtual
```

## Instalación

1. **Crear entorno virtual:**
```bash
python -m venv venv
source venv/bin/activate  # En macOS/Linux
```

2. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

## Uso

### Ejecutar Análisis Completo
```bash
python main.py
```

### Archivos Generados
- `comparacion_baseline_simplificada.png`: Visualizaciones comparativas
- `estadisticas_comparativas_simplificadas.csv`: Datos estadísticos
- `reporte_baseline_simplificado.txt`: Reporte en texto

## Funcionalidad

La aplicación realiza:
1. **Carga** de 3 archivos EEG específicos
2. **Comparación** entre baseline y condiciones
3. **Visualización** de señales y distribuciones
4. **Análisis estadístico** comparativo
5. **Generación** de reportes simples

## Dependencias

- **numpy**: Procesamiento de datos
- **matplotlib**: Visualizaciones
- **pandas**: Análisis de datos
- **scipy**: Funciones estadísticas

## Resultados

La aplicación genera una figura con 6 gráficos:
- Señales originales (baseline vs condiciones)
- Distribuciones de amplitudes
- Tabla de estadísticas comparativas
- Diferencias respecto al baseline

## Autor
Análisis EEG - 2025 # analisis_de_datos_cientificos_y_geograficos
