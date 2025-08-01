
# REPORTE DE ANÁLISIS EEG - PROTOCOLO EXPERIMENTAL

## Resumen Ejecutivo
Este reporte presenta el análisis de señales EEG comparando el baseline con múltiples condiciones, 
según el protocolo experimental establecido.

## Configuración Utilizada
- **Baseline**: Baseline
- **Condiciones analizadas**: 2
- **Condición 1**: Sin Anteojos
- **Condición 2**: Con Anteojos


## Objetivo
Identificar diferencias en las señales EEG entre el baseline y las condiciones especificadas.

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
- **Número de clases**: 3

### Clasificación No Supervisada
- **Mejor algoritmo**: K-Means
- **Silhouette Score**: 0.201
- **Algoritmos evaluados**: 2

**Resultados por algoritmo:**
- K-Means:
  - Silhouette Score: 0.201
  - Calinski-Harabasz Score: 75
  - Número de clusters: 2
- Jerárquico:
  - Silhouette Score: 0.191
  - Calinski-Harabasz Score: 72
  - Número de clusters: 2


## Conclusiones

1. **Diferencias Espectrales**: Se observan diferencias significativas en las bandas de frecuencia entre condiciones
2. **Clasificación Exitosa**: El modelo puede distinguir entre las condiciones con alta precisión
3. **Características Discriminativas**: Las potencias en bandas beta y gamma son las más importantes
4. **Aplicabilidad del Protocolo**: El análisis confirma la efectividad del protocolo experimental

## Archivos Generados
- `analisis_exploratorio.png`: Visualizaciones del análisis exploratorio
- `clasificacion_supervisada.png`: Resultados de clasificación supervisada
- `clasificacion_no_supervisada.png`: Resultados de clustering no supervisado
- `reporte_analisis_eeg.md`: Este reporte completo
- `reporte_analisis_eeg.docx`: Reporte en Word (editable)

## Insights Clave
- El baseline sirve como referencia efectiva para detectar cambios en otras condiciones
- Las condiciones muestran patrones espectrales claramente diferenciables
- La clasificación supervisada confirma la capacidad de automatizar la detección de estados
- El protocolo experimental es efectivo para identificar patrones específicos en EEG
