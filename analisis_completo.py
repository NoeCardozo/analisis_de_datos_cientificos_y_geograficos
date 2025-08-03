"""
==================
Final Assignment
==================

Todos los archivos están subidos a campus.
El largo de los registros es entre 10 y 11 minutos
Fs = 512

FECHA DE ENTREGA: 03/08/2025 23:59.59 GMT-3


|---- BASELINE --------|
|---- PESTANEO ------|
|---- OJOS CERRADOS ----|
|---- MIRAR C/ANTEOJOS ------|
|---- MIRAR S/ANTEOJOS ------|
|---- ESCUCHANDO ESPAÑOL ----|
|---- ESCUCHANDO INGLES ------|

* Baseline: esta parte la pueden utilizar para tener ejemplos negativos de cualquier cosa que deseen detectar.  Por 
ejemplo si quieren detectar que algo cambia cuando hay "imaginación en colores violeta", extraen features de ese momento y de
este e intentan armar un clasificador.
* Pestaneos: los pestaneos son eventos temporales que pueden ser detectados directamente en la señal.
* Ojos cerrados: puede aparecer un aumento de la señal en la banda alfa, la potencia de 10 Hz.
* Mirar s/anteojos o escuchando ingles: puede aparecer un aumento en señales de alta frecuencia (30,40,50 Hz).  Puede aparecer también
un aumento en la potencia de la banda alfa (x aburrimiento).

Objetivo:
El objetivo es, dado este registro implementar un análisis de estos datos, exploratorio, supervisado 
o no supervisado, para intentar identificar que es lo que el sujeto está haciendo en cada bloque.  Pueden 
intentar separar dos bloques entre sí, un bloque particular frente al BASELINE (esto es el momento cuando el sujeto
no hace nada particular).  Pueden usar una parte de dos bloques para entrenar y luego intentar predecir las otras partes.
Tienen que producir un PDF informe con gráficos/tablas breve y resumido (no más de 4 páginas)

"""

print(__doc__)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# El protocolo experimental que implementamos tiene 2 datasets:
# 1- Dataset de las señales de EEG
# 2- El video de las imágenes (de la grabación de la clase)
#
#
# La idea es tomar estos datasets y derivar de forma automática las diferentes secciones.  Esto se puede hacer en base self-supervised, es
# decir tomar los datos de algún dataset, derivar los labels para cada secciones y luego intentar implementar un clasificador multiclase.
#
# Tienen que entregar un PDF, tipo Markdown con código, gráficos y cualquier insight obtenido del dataset.

# Cargando los datos de Caro - empezamos con baseline
# Función para cargar datos EEG
def load_eeg_data(filename):
    """Carga datos EEG desde un archivo .dat"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    eeg_data = []
    timestamps = []
    counters = []
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 3:
            timestamps.append(float(parts[0]))
            counters.append(int(parts[1]))
            eeg_data.append(int(parts[2]))
    
    return np.array(eeg_data), np.array(timestamps), np.array(counters)

# Definir todas las condiciones disponibles
condiciones = {
    'baseline': 'data/dataNoelia/carodata/carobaseline.dat',
    'pestaneos': 'data/dataNoelia/carodata/caropestaneos.dat',
    'ojos_cerrados': 'data/dataNoelia/carodata/caroojoscerrados.dat',
    'mirar_con_anteojos': 'data/dataNoelia/carodata/caromirarconanteojos.dat',
    'mirar_sin_anteojos': 'data/dataNoelia/carodata/caromirarsinanteojos.dat',
    'espaniol': 'data/dataNoelia/carodata/caroespaniol.dat',
    'english': 'data/dataNoelia/carodata/caroenglish.dat'
}

# Cargar todos los datos
datos_eeg = {}
for nombre, archivo in condiciones.items():
    try:
        eeg, timestamps, counters = load_eeg_data(archivo)
        datos_eeg[nombre] = {
            'eeg': eeg,
            'timestamps': timestamps,
            'counters': counters
        }
        print(f'✓ {nombre}: {len(eeg)} muestras, duración: {(timestamps[-1] - timestamps[0]):.1f}s')
    except Exception as e:
        print(f'✗ Error cargando {nombre}: {e}')

print(f'\nTotal de condiciones cargadas: {len(datos_eeg)}')

# Crear figura con múltiples subplots para comparar todas las condiciones
fig, axes = plt.subplots(4, 2, figsize=(20, 16))
axes = axes.flatten()

# Colores para cada condición
colores = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']

# Gráfica 1: Comparación de todas las señales (primeros 10 segundos)
ax1 = axes[0]
for i, (nombre, datos) in enumerate(datos_eeg.items()):
    timestamps_norm = datos['timestamps'] - datos['timestamps'][0]
    first_10_sec = timestamps_norm <= 10
    ax1.plot(timestamps_norm[first_10_sec], datos['eeg'][first_10_sec], 
             color=colores[i], linewidth=1, alpha=0.8, label=nombre.replace('_', ' ').title())
ax1.set_xlabel('Tiempo (segundos)')
ax1.set_ylabel('Amplitud EEG')
ax1.set_title('Comparación: Primeros 10 segundos de todas las condiciones')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(True, alpha=0.3)

# Gráficas individuales para cada condición
for i, (nombre, datos) in enumerate(datos_eeg.items()):
    if i + 1 < len(axes):
        ax = axes[i + 1]
        timestamps_norm = datos['timestamps'] - datos['timestamps'][0]
        
        # Mostrar solo los primeros 20 segundos para mejor visualización
        first_20_sec = timestamps_norm <= 20
        ax.plot(timestamps_norm[first_20_sec], datos['eeg'][first_20_sec], 
                color=colores[i], linewidth=0.8, alpha=0.9)
        ax.set_xlabel('Tiempo (segundos)')
        ax.set_ylabel('Amplitud EEG')
        ax.set_title(f'{nombre.replace("_", " ").title()}')
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Estadísticas comparativas
print("\n=== ESTADÍSTICAS COMPARATIVAS ===")
print(f"{'Condición':<20} {'Media':<10} {'Std':<10} {'Min':<8} {'Max':<8} {'Duración':<10}")
print("-" * 70)
for nombre, datos in datos_eeg.items():
    eeg = datos['eeg']
    timestamps = datos['timestamps']
    duracion = timestamps[-1] - timestamps[0]
    print(f"{nombre.replace('_', ' ').title():<20} {np.mean(eeg):<10.2f} {np.std(eeg):<10.2f} "
          f"{np.min(eeg):<8} {np.max(eeg):<8} {duracion:<10.1f}s")

# Generar informe automáticamente
print("\n" + "="*50)
print("GENERANDO INFORME EN WORD...")
print("="*50)

try:
    # Importar funciones del generador de informe
    from generar_informe import generar_informe_word
    
    # Generar el informe
    output_file = generar_informe_word(datos_eeg)
    print(f"✅ Informe generado exitosamente: {output_file}")
    print("📄 El informe incluye:")
    print("   - Análisis comparativo de todas las condiciones")
    print("   - Gráficos de señales EEG")
    print("   - Estadísticas descriptivas")
    print("   - Conclusiones y recomendaciones")
    
except ImportError:
    print("❌ Error: No se pudo importar el generador de informe")
    print("   Asegúrate de que el archivo 'generar_informe.py' esté en el directorio")
except Exception as e:
    print(f"❌ Error generando informe: {e}")
    print("   Verifica que tienes instalado python-docx: pip install python-docx")

print("\n" + "="*50)
print("ANÁLISIS COMPLETADO")
print("="*50)