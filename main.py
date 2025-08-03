"""
==================
Ejecutor de Análisis EEG Completo
==================

Script principal que ejecuta el análisis completo de datos EEG y genera
el informe en Word automáticamente.

Autor: Noelia Cardozo
Fecha: 2025
"""

import sys
import os
import subprocess

def main():
    """Función principal que ejecuta el análisis completo"""
    print("="*60)
    print("ANÁLISIS COMPLETO DE DATOS EEG")
    print("Autor: Noelia Cardozo")
    print("="*60)
    
    # Verificar que los archivos necesarios existen
    archivos_requeridos = [
        'main.py',
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
    
    # Ejecutar el análisis principal
    try:
        # Ejecutar main.py usando subprocess
        print("\n" + "="*50)
        print("EJECUTANDO ANÁLISIS PRINCIPAL...")
        print("="*50)
        
        # Ejecutar main.py como proceso separado
        result = subprocess.run([sys.executable, 'analisis_completo.py'], 
                              capture_output=True, text=True)
        
        # Mostrar la salida
        print(result.stdout)
        if result.stderr:
            print("Errores:", result.stderr)
        
        print("\n" + "="*50)
        print("✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
        print("="*50)
        print("📁 Archivos generados:")
        print("   - Informe_EEG_Analisis.docx")
        print("   - imagenes_informe/comparacion_señales.png")
        print("   - imagenes_informe/estadisticas_comparativas.png")
        
    except Exception as e:
        print(f"\n❌ Error durante la ejecución: {e}")
        print("Verifica que tienes todas las dependencias instaladas:")
        print("   pip install pandas numpy matplotlib python-docx")
        return
    
    print("\n🎉 ¡Análisis completado! El informe está listo para revisar.")

if __name__ == "__main__":
    main() 