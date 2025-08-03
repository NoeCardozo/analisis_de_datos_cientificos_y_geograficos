
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.signal import butter, lfilter, welch
from scipy.fft import rfft, rfftfreq
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

def load_eeg_data(filename):
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
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_lowpass(cutoff, fs, order=5):
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

def moving_average(x, w=5):
    """Filtro temporal media móvil de ventana w"""
    return np.convolve(x, np.ones(w) / w, mode="same")


def aplicar_filtros_eeg(eeg, fs):
    """Aplica múltiples filtros para mejorar la calidad de la señal EEG"""
    
        # 0. Filtro temporal media móvil (suavizado)
    eeg_ma = moving_average(eeg, w=5)

    # 1. Filtro pasabajos para eliminar ruido de alta frecuencia (>50 Hz)
    eeg_lowpass = butter_lowpass_filter(eeg_ma, cutoff=50, fs=fs, order=4)
    
    # 2. Filtro pasabanda para la banda alfa (8-13 Hz) - importante para EEG
    eeg_alpha = butter_bandpass_filter(eeg_lowpass, lowcut=8, highcut=13, fs=fs, order=4)
    
    # 3. Filtro pasabanda para la banda beta (13-30 Hz)
    eeg_beta = butter_bandpass_filter(eeg_lowpass, lowcut=13, highcut=30, fs=fs, order=4)
    
    # 4. Filtro pasabanda para la banda theta (4-8 Hz)
    eeg_theta = butter_bandpass_filter(eeg_lowpass, lowcut=4, highcut=8, fs=fs, order=4)
    
    return {
        'original': eeg,
        'ma': eeg_ma,
        'lowpass': eeg_lowpass,
        'alpha': eeg_alpha,
        'beta': eeg_beta,
        'theta': eeg_theta
    }

# -----------------------------------------------------------------------------
#  Funciones de extracción de características EEG
# -----------------------------------------------------------------------------
import math

def crest_factor(x):
    """Crest factor (pico/RMS)"""
    x = np.asarray(x)
    return np.max(np.abs(x)) / (np.sqrt(np.mean(x ** 2)) + 1e-12)


def hjorth(a):
    """Parámetros de Hjorth: activity, mobility, complexity"""
    a = np.asarray(a)
    first_deriv = np.diff(a)
    second_deriv = np.diff(a, 2)
    activity = np.mean(a ** 2)
    mobility = np.sqrt(np.mean(first_deriv ** 2) / (activity + 1e-12))
    complexity = (
        np.sqrt(np.mean(second_deriv ** 2) / (np.mean(first_deriv ** 2) + 1e-12)) / (mobility + 1e-12)
    )
    return activity, mobility, complexity


def pfd(a):
    """Petrosian Fractal Dimension"""
    a = np.asarray(a)
    diff = np.diff(a)
    prod = diff[1:] * diff[:-1]
    n_delta = np.sum(prod < 0)
    n = len(a)
    return math.log(n) / (math.log(n) + math.log(n / (n + 0.4 * n_delta + 1e-12)))


def band_power(eeg, fs, fmin, fmax):
    """Potencia integrada en banda [fmin, fmax]"""
    freqs, pxx = welch(eeg, fs, nperseg=min(1024, len(eeg)))
    idx = (freqs >= fmin) & (freqs <= fmax)
    return np.trapz(pxx[idx], freqs[idx])


from scipy.signal import find_peaks, detrend, welch
from scipy import stats

def compute_features(eeg, fs):
    """Devuelve diccionario con características temporales y de frecuencia"""
    ptp_val = np.ptp(eeg)
    rms_val = np.sqrt(np.mean(eeg ** 2))
    cf_val = crest_factor(eeg)
    activity, mobility, complexity = hjorth(eeg)
    pfd_val = pfd(eeg)
    delta_power = band_power(eeg, fs, 1, 4)
    theta_power = band_power(eeg, fs, 4, 8)
    alpha_power = band_power(eeg, fs, 8, 13)
    beta_power = band_power(eeg, fs, 13, 30)
    # --- Outliers (z-score > 3)
    z = (eeg - np.mean(eeg)) / (np.std(eeg) + 1e-12)
    outliers_z3 = np.sum(np.abs(z) > 3)
    # --- Peaks por ventana (umbral 200 µV)
    peaks, _ = find_peaks(eeg, height=200)
    n_peaks = len(peaks)
    return {
        'ptp': ptp_val,
        'rms': rms_val,
        'crest_factor': cf_val,
        'hjorth_activity': activity,
        'hjorth_mobility': mobility,
        'hjorth_complexity': complexity,
        'pfd': pfd_val,
        'power_delta': delta_power,
        'power_theta': theta_power,
        'power_alpha': alpha_power,
        'power_beta': beta_power,
        'outliers_z3': outliers_z3,
        'n_peaks': n_peaks,
    }

# -----------------------------------------------------------------------------

def crear_grafico_ma(datos_eeg, save_path):
    """Gráfico comparación media móvil vs original (primeros 10s)"""
    fig, axes = plt.subplots(4, 2, figsize=(15, 12))
    axes = axes.flatten()
    colores = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']

    # Panel global
    ax_global = axes[0]
    for i, (nombre, datos) in enumerate(datos_eeg.items()):
        t_norm = datos['timestamps'] - datos['timestamps'][0]
        mask = t_norm <= 10
        ax_global.plot(t_norm[mask], datos['eeg'][mask], color='black', linewidth=0.8, alpha=0.7)
        ax_global.plot(t_norm[mask], datos['eeg_filtrado']['ma'][mask], color=colores[i], linewidth=1, alpha=0.8,
                       label=nombre.replace('_', ' ').title())
    ax_global.set_title('Comparación Media Móvil (Primeros 10s)')
    ax_global.set_xlabel('Tiempo (s)')
    ax_global.set_ylabel('Amplitud EEG')
    ax_global.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax_global.grid(True, alpha=0.3)

    # Paneles individuales
    for i, (nombre, datos) in enumerate(datos_eeg.items()):
        if i + 1 < len(axes):
            ax = axes[i + 1]
            t_norm = datos['timestamps'] - datos['timestamps'][0]
            mask = t_norm <= 20
            ax.plot(t_norm[mask], datos['eeg'][mask], color='black', linewidth=0.8, alpha=0.7, label='Original')
            ax.plot(t_norm[mask], datos['eeg_filtrado']['ma'][mask], color=colores[i], linewidth=1, alpha=0.9, label='MA')
            ax.set_title(f'{nombre.replace("_", " ").title()} - Original vs MA')
            ax.set_xlabel('Tiempo (s)')
            ax.set_ylabel('Amplitud EEG')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path


def crear_grafico_comparativo_filtrado(datos_eeg, save_path):
    """Crea gráfico comparativo de todas las condiciones con filtros aplicados"""
    fig, axes = plt.subplots(4, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    colores = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']
    
    # Gráfica 1: Comparación de señales filtradas (primeros 10 segundos)
    # También generaremos comparación con media móvil
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
                    color='black', linewidth=1.0, alpha=0.8, label='Original')
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
            # Convertir PSD a dB para resaltar mejor las diferencias entre condiciones
            psd_original_db = 10 * np.log10(psd_original + 1e-12)
            psd_lowpass_db = 10 * np.log10(psd_lowpass + 1e-12)
            psd_alpha_db   = 10 * np.log10(psd_alpha + 1e-12)

            ax.plot(freqs, psd_original_db, color='gray', alpha=0.6, label='Original', linewidth=0.8)
            ax.plot(freqs, psd_lowpass_db, color=colores[i], alpha=0.9, label='Filtrada', linewidth=1)
            ax.plot(freqs, psd_alpha_db, color='red', alpha=0.7, label='Banda Alfa', linewidth=1)
            
            ax.set_xlabel('Frecuencia (Hz)')
            ax.set_ylabel('Potencia (dB/Hz)')
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
            
            print(f'[OK] {nombre}: {len(eeg)} muestras, fs={fs:.1f} Hz')
            
        except Exception as e:
            print(f'[ERROR] Error cargando {nombre}: {e}')
    
    if not datos_eeg:
        print("Error: No se pudieron cargar datos EEG")
        return
    
    print(f"\nTotal de condiciones cargadas: {len(datos_eeg)}")
    
    # Crear directorio para imágenes si no existe
    if not os.path.exists('imagenes_informe'):
        os.makedirs('imagenes_informe')
    
    # Generar gráficos con filtros aplicados
    print("\nGenerando gráficos con filtros aplicados...")
    
    # Gráfico comparativo media móvil
    grafico_ma_path = crear_grafico_ma(datos_eeg, 'imagenes_informe/comparacion_ma.png')
    print(f"[OK] Gráfico media móvil: {grafico_ma_path}")

    # Gráfico comparativo con filtros
    grafico_filtrado_path = crear_grafico_comparativo_filtrado(datos_eeg, 'imagenes_informe/comparacion_filtrada.png')
    print(f"[OK] Gráfico comparativo filtrado: {grafico_filtrado_path}")
    
    # Gráfico de espectro de frecuencia
    grafico_espectro_path = crear_grafico_espectro_frecuencia(datos_eeg, 'imagenes_informe/espectro_frecuencia.png')
    print(f"[OK] Gráfico de espectro: {grafico_espectro_path}")
    
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
    
    # ------------------------------------------------------------------
    # Paso extra: Clasificación automática de la condición EEG
    # ------------------------------------------------------------------
    print("\n=== CLASIFICACIÓN AUTOMÁTICA DE CONDICIONES ===")

    feature_rows = []
    feature_labels = []

    for nombre, datos in datos_eeg.items():
        eeg_lp = datos['eeg_filtrado']['lowpass']
        fs_curr = datos['fs']
        ventana = 512  # 1 segundo por ventana a 512 Hz
        pasos = range(0, len(eeg_lp) - ventana, ventana)
        for start in pasos:
            segmento = eeg_lp[start:start + ventana]
            feats = compute_features(segmento, fs_curr)
            feature_rows.append(list(feats.values()))
            feature_labels.append(nombre)

    if len(set(feature_labels)) > 1:
        X = np.array(feature_rows)
        y = np.array(feature_labels)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Exactitud global: {acc*100:.2f}% con {len(y_test)} muestras de prueba")

        cm = confusion_matrix(y_test, y_pred, labels=list(condiciones.keys()))
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(cm, display_labels=[c.replace('_',' ') for c in condiciones.keys()])
        disp.plot(ax=ax_cm, cmap='Blues', xticks_rotation=45)
        plt.tight_layout()
        cm_path = 'imagenes_informe/confusion_matrix.png'
        fig_cm.savefig(cm_path, dpi=300)
        plt.close(fig_cm)
        print(f"Matriz de confusión guardada en {cm_path}")
        # Guardar métrica de exactitud para incluir en informe
        import json, pathlib
        metrics_out = pathlib.Path('imagenes_informe/classification_metrics.json')
        with metrics_out.open('w') as jf:
            json.dump({'accuracy': float(acc)}, jf)
        print(f"Métricas guardadas en {metrics_out}")
    else:
        print("No hay suficientes clases para entrenar un clasificador.")

    # ------------------------------------------------------------------
    print("\n" + "="*50)
    print("ANÁLISIS CON FILTROS Y CLASIFICACIÓN COMPLETADOS")
    print("="*50)
    
    # Retornar datos para el generador de informe
    return datos_eeg

if __name__ == "__main__":
    main()