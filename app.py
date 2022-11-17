"""
equisitos de diseño del sistema:
• No es válido utilizar librerías que implementen la K-NN
. Este debe ser unaimplementación tuya de todos los pasos del algoritmo K-NN, incluyendo
el cálculo de las métricas deberá ser tu propia implementación.• Debe incluir al menos el
llamado de 2 métodos:1. Un método para cargar el conjunto de datos de entrenamiento a memoria
para que se pueda leer su contenidomientras lleguen datos nuevos a clasificar. Un ejemplo para invocar este método
sería:Objetos[] leerDatosEntrenamiento(nombre_archivo)

2. Un método que estime la distancia del punto a clasificar con todos los puntos de entrenamiento. Este métododebe recibir al menos 2 parámetos (el número de vecinos cercanos a utilizar (k) y el punto a clasificar.
Si requieres incluir parámetros adicionales, lo dejo a tu consideración y deberás explicarlo.



Requisitos:• Utilizar el 30% del conjunto de datos que se te dará para usarlo como conuntode datos de entrenamiento
• El otro 70% lo utilizarás para la prueba. Durante la prueba, se ejecuta el K-nn ycada punto de este 70% lo clasifica
en una de las clases.

• Una vez que termine la prueba, el sistema debe estimar la exactitud, laprecisión, la exhaustividad
y F-score; y mostrarlos en pantalla.
"""
# Importar librerías
import math
import numpy as np


def distanciaEuclidiana(puntoA, puntoB):
    # Ciclo for para generar un string que es equivalente a la distancia euclidiana de n dimensiones
    string = ""
    for i in range(len(puntoA)):
        string += "(" + str(puntoA[i]) + "-" + str(puntoB[i]) + ")**2 + "
    string = string[:-3]  # Aqui se quita el ultimo "+"
    distancia = eval(string)  # Aqui se evalua el string
    print("String: ", string)

    # Aqui usamos una formula en vez de un ciclo porque es mas rapido, y tiene una complejidad de O(1) en vez de O(n)
    return distancia


print(distanciaEuclidiana([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [
      11, 12, 13, 14, 15, 16, 17, 18, 19, 20]))
