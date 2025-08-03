# Análisis de Datos EEG con Filtros Espectrales

## 📋 Descripción del Proyecto

Este proyecto realiza un análisis completo de señales EEG registradas bajo 7 condiciones experimentales diferentes, aplicando filtros espectrales para mejorar la calidad de los datos y generar un informe profesional en Word.

**Autor:** Noelia Cardozo  
**Fecha:** 2025

## 🧠 Condiciones Experimentales Analizadas

1. **Baseline** - Estado de reposo sin actividad específica
2. **Pestañeos** - Eventos transitorios de pestañeo
3. **Ojos cerrados** - Actividad con ojos cerrados (aumento de actividad alfa)
4. **Mirar con anteojos** - Actividad visual con anteojos
5. **Mirar sin anteojos** - Actividad visual sin anteojos
6. **Escuchando español** - Procesamiento auditivo en español
7. **Escuchando inglés** - Procesamiento auditivo en inglés

## 📁 Estructura del Proyecto

```
python-project/
├── main.py                          # 🚀 Script principal (ejecuta todo)
├── analisis_completo.py             # 📊 Análisis con filtros espectrales
├── generar_informe.py               # 📄 Generador de informe en Word
├── filter_spectral.py               # 🔧 Filtros espectrales (referencia)
├── data/
│   └── dataNoelia/
│       └── carodata/
│           ├── carobaseline.dat
│           ├── caropestaneos.dat
│           ├── caroojoscerrados.dat
│           ├── caromirarconanteojos.dat
│           ├── caromirarsinanteojos.dat
│           ├── caroespaniol.dat
│           └── caroenglish.dat
├── imagenes_informe/                # 📈 Imágenes generadas
│   ├── comparacion_señales.png
│   ├── estadisticas_comparativas.png
│   ├── comparacion_filtrada.png
│   └── espectro_frecuencia.png
├── Informe_EEG_Analisis.docx        # 📋 Informe final en Word
└── README_EEG_Analysis.md           # 📖 Este archivo
```

## 🚀 Instalación y Configuración

### 1. Requisitos Previos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### 2. Instalación de Dependencias

```bash
# Crear entorno virtual (recomendado)
python -m venv venv

# Activar entorno virtual
# En Windows:
venv\Scripts\activate
# En macOS/Linux:
source venv/bin/activate

# Instalar dependencias
pip install pandas numpy matplotlib python-docx scipy
```

### 3. Verificar Instalación

```bash
python -c "import pandas, numpy, matplotlib, docx, scipy; print('✅ Todas las dependencias instaladas correctamente')"
```

## 📊 Uso del Proyecto

### 🎯 Opción Principal (Recomendada)

```bash
python main.py
```

Este comando ejecuta automáticamente:
1. **PASO 1**: Análisis completo con filtros espectrales
2. **PASO 2**: Generación del informe en Word

### 🔧 Opciones Individuales

#### Solo Análisis con Filtros
```bash
python analisis_completo.py
```

#### Solo Generar Informe
```bash
python generar_informe.py
```

## 🔧 Filtros Espectrales Aplicados

### Tipos de Filtros
1. **Filtro Pasabajos (50 Hz)** - Elimina ruido de alta frecuencia
2. **Filtro Pasabanda Alfa (8-13 Hz)** - Aísla actividad alfa característica
3. **Filtro Pasabanda Beta (13-30 Hz)** - Aísla actividad beta
4. **Filtro Pasabanda Theta (4-8 Hz)** - Aísla actividad theta

### Beneficios
- ✅ **Mejor SNR** (Signal-to-Noise Ratio)
- ✅ **Reducción de artefactos** y ruido
- ✅ **Identificación clara** de bandas de frecuencia
- ✅ **Análisis más preciso** de patrones EEG

## 📈 Características del Análisis

### Estadísticas Calculadas
- **Media** de amplitud EEG (original y filtrada)
- **Desviación estándar** (variabilidad)
- **Valores mínimo y máximo**
- **Rango** de amplitud
- **Duración** del registro
- **Frecuencia de muestreo** estimada
- **Potencia en bandas** alfa, beta y theta

### Visualizaciones Generadas
1. **Comparación de señales** - Primeros 10 segundos de todas las condiciones
2. **Gráficos individuales** - 20 segundos de cada condición
3. **Análisis estadístico** - 4 gráficos comparativos
4. **Comparación filtrada** - Original vs filtrada
5. **Espectro de frecuencia** - Análisis espectral completo

## 📄 Informe Generado

El informe en Word (`Informe_EEG_Analisis.docx`) incluye:

### Secciones Principales
1. **Información del Proyecto** - Metadatos y descripción
2. **Resumen Ejecutivo** - Descripción general del análisis
3. **Estadísticas Generales** - Tabla completa de estadísticas
4. **Análisis de Resultados** - Interpretación de los datos
5. **Visualización de Datos** - Gráficos embebidos
6. **Análisis con Filtros Espectrales** - Nueva sección con filtros
7. **Conclusiones** - Hallazgos principales
8. **Recomendaciones** - Sugerencias para análisis futuro

### Hallazgos Principales
- **Ojos cerrados**: Menor variabilidad (actividad alfa aumentada)
- **Pestañeos**: Mayor variabilidad (eventos transitorios)
- **Condiciones auditivas**: Patrones similares entre español e inglés
- **Actividad visual**: Diferencias entre usar o no anteojos
- **Mejora con filtros**: Reducción significativa de ruido

## 🔧 Personalización

### Modificar Condiciones
Para agregar o cambiar condiciones, edita el diccionario `condiciones` en `analisis_completo.py`:

```python
condiciones = {
    'nueva_condicion': 'ruta/al/archivo.dat',
    # ... otras condiciones
}
```

### Cambiar Configuración de Filtros
Modifica los parámetros en `aplicar_filtros_eeg()` en `analisis_completo.py`:
- Frecuencia de corte pasabajos: `cutoff=50`
- Bandas de frecuencia: `lowcut=8, highcut=13`
- Orden del filtro: `order=4`

### Personalizar Informe
Edita la función `generar_informe_word()` en `generar_informe.py` para:
- Cambiar el título
- Agregar secciones
- Modificar el formato
- Incluir análisis adicionales

## 🐛 Solución de Problemas

### Error: "No module named 'scipy'"
```bash
pip install scipy
```

### Error: "No module named 'docx'"
```bash
pip install python-docx
```

### Error: "No se pudieron cargar datos EEG"
- Verifica que los archivos .dat estén en la ruta correcta
- Asegúrate de que los archivos no estén corruptos
- Verifica el formato de los datos (timestamp, counter, eeg, attention, meditation)

### Error: "No se pudo generar el informe"
- Verifica que tienes permisos de escritura en el directorio
- Asegúrate de que python-docx esté instalado correctamente
- Verifica que hay datos cargados antes de generar el informe

## 📚 Dependencias

- **pandas**: Manipulación y análisis de datos
- **numpy**: Cálculos numéricos
- **matplotlib**: Generación de gráficos
- **python-docx**: Creación de documentos Word
- **scipy**: Filtros espectrales y procesamiento de señales

## 🎯 Flujo de Trabajo

```
1. main.py
   ↓
2. analisis_completo.py
   ├── Carga datos EEG
   ├── Aplica filtros espectrales
   ├── Genera gráficos
   └── Calcula estadísticas
   ↓
3. generar_informe.py
   ├── Crea documento Word
   ├── Incluye gráficos
   ├── Agrega análisis
   └── Genera informe final
```

## 🤝 Contribuciones

Para contribuir al proyecto:
1. Fork el repositorio
2. Crea una rama para tu feature
3. Realiza los cambios
4. Envía un pull request

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver el archivo LICENSE para más detalles.

## 📞 Contacto

**Autor:** Noelia Cardozo  
**Email:** [tu-email@ejemplo.com]  
**Fecha:** 2025

---

**Nota:** Este proyecto fue desarrollado para el análisis de datos EEG como parte de una tarea académica. Los datos utilizados son propiedad de Noelia Cardozo. 