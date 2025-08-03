"""
==================
Análisis Completo de Datos EEG con Filtros Espectrales
==================

Script principal que ejecuta el análisis completo de datos EEG con filtros
espectrales para suavizar y mejorar la calidad de los datos.

Autor: Noelia Cardozo
Fecha: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.signal import butter, lfilter, welch
from scipy.fft import rfft, rfftfreq
import os

# Configurar matplotlib para español
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

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

def butter_bandpass(lowcut, highcut, fs, order=5):
    """Diseña un filtro pasabanda Butterworth"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Aplica un filtro pasabanda Butterworth a los datos"""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_lowpass(cutoff, fs, order=5):
    """Diseña un filtro pasabajos Butterworth"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    """Aplica un filtro pasabajos Butterworth a los datos"""
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def calcular_estadisticas(eeg, timestamps):
    """Calcula estadísticas descriptivas de la señal EEG"""
    duracion = timestamps[-1] - timestamps[0]
    fs_estimada = len(eeg) / duracion
    
    return {
        'media': np.mean(eeg),
        'mediana': np.median(eeg),
        'std': np.std(eeg),
        'min': np.min(eeg),
        'max': np.max(eeg),
        'duracion': duracion,
        'fs': fs_estimada,
        'rango': np.max(eeg) - np.min(eeg)
    }

def aplicar_filtros_eeg(eeg, fs):
    """Aplica múltiples filtros para mejorar la calidad de la señal EEG"""
    
    # 1. Filtro pasabajos para eliminar ruido de alta frecuencia (>50 Hz)
    eeg_lowpass = butter_lowpass_filter(eeg, cutoff=50, fs=fs, order=4)
    
    # 2. Filtro pasabanda para la banda alfa (8-13 Hz) - importante para EEG
    eeg_alpha = butter_bandpass_filter(eeg_lowpass, lowcut=8, highcut=13, fs=fs, order=4)
    
    # 3. Filtro pasabanda para la banda beta (13-30 Hz)
    eeg_beta = butter_bandpass_filter(eeg_lowpass, lowcut=13, highcut=30, fs=fs, order=4)
    
    # 4. Filtro pasabanda para la banda theta (4-8 Hz)
    eeg_theta = butter_bandpass_filter(eeg_lowpass, lowcut=4, highcut=8, fs=fs, order=4)
    
    return {
        'original': eeg,
        'lowpass': eeg_lowpass,
        'alpha': eeg_alpha,
        'beta': eeg_beta,
        'theta': eeg_theta
    }

def crear_grafico_comparativo_filtrado(datos_eeg, save_path):
    """Crea gráfico comparativo de todas las condiciones con filtros aplicados"""
    fig, axes = plt.subplots(4, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    colores = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']
    
    # Gráfica 1: Comparación de señales filtradas (primeros 10 segundos)
    ax1 = axes[0]
    for i, (nombre, datos) in enumerate(datos_eeg.items()):
        timestamps_norm = datos['timestamps'] - datos['timestamps'][0]
        first_10_sec = timestamps_norm <= 10
        # Usar señal filtrada con pasabajos
        ax1.plot(timestamps_norm[first_10_sec], datos['eeg_filtrado']['lowpass'][first_10_sec], 
                 color=colores[i], linewidth=1, alpha=0.8, label=nombre.replace('_', ' ').title())
    ax1.set_xlabel('Tiempo (segundos)')
    ax1.set_ylabel('Amplitud EEG (Filtrada)')
    ax1.set_title('Comparación: Señales EEG Filtradas (Primeros 10s)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Gráficas individuales para cada condición (original vs filtrada)
    for i, (nombre, datos) in enumerate(datos_eeg.items()):
        if i + 1 < len(axes):
            ax = axes[i + 1]
            timestamps_norm = datos['timestamps'] - datos['timestamps'][0]
            
            # Mostrar solo los primeros 20 segundos
            first_20_sec = timestamps_norm <= 20
            
            # Graficar señal original y filtrada
            ax.plot(timestamps_norm[first_20_sec], datos['eeg'][first_20_sec], 
                    color='gray', linewidth=0.5, alpha=0.6, label='Original')
            ax.plot(timestamps_norm[first_20_sec], datos['eeg_filtrado']['lowpass'][first_20_sec], 
                    color=colores[i], linewidth=1, alpha=0.9, label='Filtrada')
            
            ax.set_xlabel('Tiempo (segundos)')
            ax.set_ylabel('Amplitud EEG')
            ax.set_title(f'{nombre.replace("_", " ").title()} - Original vs Filtrada')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path

def crear_grafico_espectro_frecuencia(datos_eeg, save_path):
    """Crea gráfico del espectro de frecuencia para todas las condiciones"""
    fig, axes = plt.subplots(4, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    colores = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']
    
    # Gráfica 1: Comparación de espectros de frecuencia
    ax1 = axes[0]
    for i, (nombre, datos) in enumerate(datos_eeg.items()):
        # Calcular espectro de potencia usando Welch
        freqs, psd = welch(datos['eeg_filtrado']['lowpass'], fs=datos['fs'], nperseg=1024)
        ax1.semilogy(freqs, psd, color=colores[i], alpha=0.8, 
                    label=nombre.replace('_', ' ').title(), linewidth=1)
    
    ax1.set_xlabel('Frecuencia (Hz)')
    ax1.set_ylabel('Densidad Espectral de Potencia')
    ax1.set_title('Espectro de Frecuencia - Todas las Condiciones')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 50)  # Mostrar hasta 50 Hz
    
    # Gráficas individuales del espectro para cada condición
    for i, (nombre, datos) in enumerate(datos_eeg.items()):
        if i + 1 < len(axes):
            ax = axes[i + 1]
            
            # Calcular espectros para diferentes bandas
            freqs, psd_original = welch(datos['eeg'], fs=datos['fs'], nperseg=1024)
            freqs, psd_lowpass = welch(datos['eeg_filtrado']['lowpass'], fs=datos['fs'], nperseg=1024)
            freqs, psd_alpha = welch(datos['eeg_filtrado']['alpha'], fs=datos['fs'], nperseg=1024)
            
            ax.semilogy(freqs, psd_original, color='gray', alpha=0.6, label='Original', linewidth=0.8)
            ax.semilogy(freqs, psd_lowpass, color=colores[i], alpha=0.9, label='Filtrada', linewidth=1)
            ax.semilogy(freqs, psd_alpha, color='red', alpha=0.7, label='Banda Alfa', linewidth=1)
            
            ax.set_xlabel('Frecuencia (Hz)')
            ax.set_ylabel('Densidad Espectral')
            ax.set_title(f'{nombre.replace("_", " ").title()} - Espectro')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 50)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path

def main():
    """Función principal"""
    print("="*60)
    print("ANÁLISIS COMPLETO DE DATOS EEG CON FILTROS ESPECTRALES")
    print("Autor: Noelia Cardozo")
    print("="*60)
    
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
    
    # Cargar todos los datos y aplicar filtros
    print("Cargando datos EEG y aplicando filtros...")
    datos_eeg = {}
    
    for nombre, archivo in condiciones.items():
        try:
            eeg, timestamps, counters = load_eeg_data(archivo)
            
            # Calcular frecuencia de muestreo
            fs = len(eeg) / (timestamps[-1] - timestamps[0])
            
            # Aplicar filtros
            eeg_filtrado = aplicar_filtros_eeg(eeg, fs)
            
            datos_eeg[nombre] = {
                'eeg': eeg,
                'timestamps': timestamps,
                'counters': counters,
                'fs': fs,
                'eeg_filtrado': eeg_filtrado
            }
            
            print(f'✓ {nombre}: {len(eeg)} muestras, fs={fs:.1f} Hz')
            
        except Exception as e:
            print(f'✗ Error cargando {nombre}: {e}')
    
    if not datos_eeg:
        print("Error: No se pudieron cargar datos EEG")
        return
    
    print(f"\nTotal de condiciones cargadas: {len(datos_eeg)}")
    
    # Crear directorio para imágenes si no existe
    if not os.path.exists('imagenes_informe'):
        os.makedirs('imagenes_informe')
    
    # Generar gráficos con filtros aplicados
    print("\nGenerando gráficos con filtros aplicados...")
    
    # Gráfico comparativo con filtros
    grafico_filtrado_path = crear_grafico_comparativo_filtrado(datos_eeg, 'imagenes_informe/comparacion_filtrada.png')
    print(f"✓ Gráfico comparativo filtrado: {grafico_filtrado_path}")
    
    # Gráfico de espectro de frecuencia
    grafico_espectro_path = crear_grafico_espectro_frecuencia(datos_eeg, 'imagenes_informe/espectro_frecuencia.png')
    print(f"✓ Gráfico de espectro: {grafico_espectro_path}")
    
    # Estadísticas comparativas (usando datos filtrados)
    print("\n=== ESTADÍSTICAS COMPARATIVAS (DATOS FILTRADOS) ===")
    print(f"{'Condición':<20} {'Media':<10} {'Std':<10} {'Min':<8} {'Max':<8} {'Duración':<10}")
    print("-" * 70)
    
    for nombre, datos in datos_eeg.items():
        eeg_filtrado = datos['eeg_filtrado']['lowpass']  # Usar señal filtrada
        timestamps = datos['timestamps']
        duracion = timestamps[-1] - timestamps[0]
        print(f"{nombre.replace('_', ' ').title():<20} {np.mean(eeg_filtrado):<10.2f} {np.std(eeg_filtrado):<10.2f} "
              f"{np.min(eeg_filtrado):<8.0f} {np.max(eeg_filtrado):<8.0f} {duracion:<10.1f}s")
    
    # Análisis de bandas de frecuencia
    print("\n=== ANÁLISIS DE BANDAS DE FRECUENCIA ===")
    print(f"{'Condición':<20} {'Potencia Alfa':<15} {'Potencia Beta':<15} {'Potencia Theta':<15}")
    print("-" * 70)
    
    for nombre, datos in datos_eeg.items():
        # Calcular potencia en cada banda
        potencia_alfa = np.mean(datos['eeg_filtrado']['alpha']**2)
        potencia_beta = np.mean(datos['eeg_filtrado']['beta']**2)
        potencia_theta = np.mean(datos['eeg_filtrado']['theta']**2)
        
        print(f"{nombre.replace('_', ' ').title():<20} {potencia_alfa:<15.2f} {potencia_beta:<15.2f} {potencia_theta:<15.2f}")
    
    print("\n" + "="*50)
    print("ANÁLISIS CON FILTROS COMPLETADO")
    print("="*50)
    
    # Retornar datos para el generador de informe
    return datos_eeg

if __name__ == "__main__":
    main()