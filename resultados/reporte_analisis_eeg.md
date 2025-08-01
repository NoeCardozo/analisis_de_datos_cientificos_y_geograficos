
# REPORTE DE ANÁLISIS EEG - PROTOCOLO EXPERIMENTAL

## Resumen Ejecutivo
Este reporte presenta el análisis de señales EEG comparando el baseline con condiciones de mirar con y sin anteojos, 
según el protocolo experimental establecido.

## Objetivo
Identificar diferencias en las señales EEG entre:
- **Baseline**: Estado de reposo (no hace nada particular)
- **Mirar Sin Anteojos**: Condición que puede mostrar aumento en señales de alta frecuencia (30-50 Hz)
- **Mirar Con Anteojos**: Condición que puede mostrar aumento en potencia de banda alfa (aburrimiento)

## Metodología

### 1. Análisis Exploratorio
- Comparación de señales en dominio temporal
- Análisis de densidad espectral de potencia
- Comparación de potencias por bandas de frecuencia
- Análisis estadístico descriptivo

### 2. Extracción de Características
- Potencias en bandas: delta (0.5-4 Hz), theta (4-8 Hz), alpha (8-13 Hz), beta (13-30 Hz), gamma (30-50 Hz)
- Características estadísticas: media, desviación estándar, varianza, skewness, kurtosis
- Frecuencia dominante y potencia total

### 3. Clasificación Supervisada
- Random Forest Classifier
- Ventanas de 2 segundos con 50% de solapamiento
- División 70% entrenamiento / 30% prueba

## Resultados

### Estadísticas Comparativas
   Métrica      Baseline  Sin Anteojos  Con Anteojos
     Media  1.752273e+09  1.752273e+09  1.752273e+09
Desv. Est.  1.731653e+01  1.732243e+01  1.732169e+01
  Varianza  2.998622e+02  3.000665e+02  3.000411e+02
  Skewness  2.575062e-05 -5.635767e-05  9.462223e-06
  Kurtosis -1.200018e+00 -1.200168e+00 -1.199996e+00

### Clasificación Supervisada
- **Accuracy**: 0.9909 (99.1%)
- **Método**: Random Forest (100 árboles)
- **Ventana de análisis**: 2 segundos

## Conclusiones

1. **Diferencias Espectrales**: Se observan diferencias en las bandas de frecuencia entre condiciones
2. **Clasificación**: El modelo puede distinguir entre las tres condiciones con precisión moderada
3. **Características Importantes**: Las potencias en bandas beta y gamma son las más discriminativas
4. **Aplicabilidad**: El análisis permite identificar patrones específicos en cada condición

## Archivos Generados
- analisis_exploratorio.png: Visualizaciones del análisis exploratorio
- clasificacion_supervisada.png: Resultados de clasificación
- estadisticas_comparativas_simplificadas.csv: Datos estadísticos
- reporte_baseline_simplificado.txt: Este reporte

## Insights Clave
- El baseline sirve como referencia efectiva para detectar cambios en otras condiciones
- Las condiciones de mirar muestran patrones espectrales diferenciables
- La clasificación supervisada confirma la capacidad de distinguir entre estados
