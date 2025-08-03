"""
==================
Análisis Principal de Datos EEG
==================

Script principal que ejecuta el análisis completo de datos EEG con filtros
espectrales y genera automáticamente el informe en Word.

Autor: Noelia Cardozo
Fecha: 2025
"""

import sys
import os
import subprocess

def main():
    """Función principal que ejecuta el análisis completo"""
    print("="*60)
    print("ANÁLISIS PRINCIPAL DE DATOS EEG")
    print("Autor: Noelia Cardozo")
    print("="*60)
    
    # Verificar que los archivos necesarios existen
    archivos_requeridos = [
        'analisis_completo.py',
        'generar_informe.py',
        'data/dataNoelia/carodata/carobaseline.dat',
        'data/dataNoelia/carodata/caropestaneos.dat',
        'data/dataNoelia/carodata/caroojoscerrados.dat',
        'data/dataNoelia/carodata/caromirarconanteojos.dat',
        'data/dataNoelia/carodata/caromirarsinanteojos.dat',
        'data/dataNoelia/carodata/caroespaniol.dat',
        'data/dataNoelia/carodata/caroenglish.dat'
    ]
    
    print("Verificando archivos necesarios...")
    for archivo in archivos_requeridos:
        if os.path.exists(archivo):
            print(f"✓ {archivo}")
        else:
            print(f"✗ {archivo} - NO ENCONTRADO")
            return
    
    print("\nTodos los archivos están presentes. Iniciando análisis...")
    
    # Ejecutar el análisis completo
    try:
        print("\n" + "="*50)
        print("PASO 1: EJECUTANDO ANÁLISIS COMPLETO...")
        print("="*50)
        
        # Ejecutar análisis_completo.py
        result = subprocess.run([sys.executable, 'analisis_completo.py'], 
                              capture_output=True, text=True)
        
        # Mostrar la salida
        print(result.stdout)
        if result.stderr:
            print("Errores:", result.stderr)
        
        print("\n" + "="*50)
        print("PASO 2: GENERANDO INFORME EN WORD...")
        print("="*50)
        
        # Ejecutar generar_informe.py
        result2 = subprocess.run([sys.executable, 'generar_informe.py'], 
                               capture_output=True, text=True)
        
        # Mostrar la salida
        print(result2.stdout)
        if result2.stderr:
            print("Errores:", result2.stderr)
        
        print("\n" + "="*50)
        print("✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
        print("="*50)
        print("📁 Archivos generados:")
        print("   - Informe_EEG_Analisis.docx")
        print("   - imagenes_informe/comparacion_señales.png")
        print("   - imagenes_informe/estadisticas_comparativas.png")
        print("   - imagenes_informe/comparacion_filtrada.png")
        print("   - imagenes_informe/espectro_frecuencia.png")
        
        print("\n🎯 Características del análisis:")
        print("   • Análisis de 7 condiciones EEG")
        print("   • Filtros espectrales aplicados")
        print("   • Análisis de bandas de frecuencia")
        print("   • Informe completo en Word")
        
    except Exception as e:
        print(f"\n❌ Error durante la ejecución: {e}")
        print("Verifica que tienes todas las dependencias instaladas:")
        print("   pip install pandas numpy matplotlib python-docx scipy")
        return
    
    print("\n🎉 ¡Análisis completado! El informe está listo para revisar.")

if __name__ == "__main__":
    main() 