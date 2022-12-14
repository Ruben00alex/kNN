
# Importar librerías
import math
import numpy as np

# Leer datos de entrenamiento


def leerDatosEntrenamiento(nombre_archivo):
    archivo = open(nombre_archivo, 'r')  # abrimos el archivo
    lineas = archivo.readlines()  # separamos las lineas del archivo
    archivo.close()
    # matriz de numpy
    objetos = np.zeros((len(lineas), len(lineas[0].split(','))+1))
    # con el tamaño de las lineas y el numero de columnas del archivo
    # Recorremos las lineas del archivo
    for x, linea in enumerate(lineas):
        linea = linea.replace('\n', '')  # quitamos el salto de linea
        atributos = linea.split(',')  # separa los atributos por coma
        # print(atributos)
        # recorrer atributos y asignarlos a indice de objeto[x][y]
        for y, atributo in enumerate(atributos):
            # print(x, y, atributo)
            objetos[x][y] = atributo
            objetos[x][5] = int(x)# se agrega el indice del punto de entrenamiento
    return objetos


datos = leerDatosEntrenamiento('participante_yh13_training.csv')
print('Datos de entrenamiento')
print(datos)


def leerDatosPrueba(nombre_archivo):
    archivo = open(nombre_archivo, 'r')
    lineas = archivo.readlines()
    archivo.close()
    objetos = np.zeros((len(lineas)-1, 6))

    for x, linea in enumerate(lineas):
        if x > 0:  # se salta la primera linea
            # linea = linea.replace('\n', '')
            atributos = linea.split(',')
            objetos[x-1][0] = atributos[1]
            objetos[x-1][1] = atributos[2]
            objetos[x-1][2] = atributos[3]
            objetos[x-1][3] = atributos[4]
            objetos[x-1][4] = atributos[5]
            # timestamp = atributos[0]
            objetos[x-1][5] = atributos[0]
        else:
            print('Se salta la primera linea')
            print(linea)
    return objetos


datosPrueba = leerDatosPrueba('participante_yh13_dataset.csv')

print("Datos de prueba:")
print(datosPrueba)
# print(datos)

#las funciones leerDatosEntrenamiento(nombre_archivo) y  leerDatosPrueba(nombre_archivo),  respectivamente, estas funciones regresan un arreglo de numpy con los datos de entrenamiento y de prueba, respectivamente. Cada fila del arreglo es un punto, y cada columna es una característica del punto. La última columna es la clase del punto. Se utilizaron 2 funciones debido a que el archivo de entrenamiento y el de prueba tienen un formato diferente.




def distanciaEuclidiana(puntoA, puntoB):
    distancia = math.sqrt((puntoA[0] - puntoB[0])**2 + (puntoA[1] - puntoB[1])
                          ** 2 + (puntoA[2] - puntoB[2])**2 + (puntoA[3] - puntoB[3])**2)
    # Aqui usamos una formula en vez de un ciclo porque es mas rapido, y tiene una complejidad de O(1) en vez de O(n)
    return distancia


# punto es un vector de 5 dimensiones, donde las 4 primeras son las características y la última es la clase que tiene el punto.

def clasificarPunto(punto, k):
    distancias = []  # lista donde se guardan las distancias de todos los puntos de entrenamiento con el punto a clasificar
    # lista donde se guardan los puntos de entrenamiento que son los k puntos mas cercanos al punto a clasificar
    puntosKMascercanos = []
    puntoClasificado = []  # lista donde se guarda el punto clasificado que se va a retornar

    # Se calcula la distancia de todos los puntos de entrenamiento con el punto a clasificar
    for x, puntoEntrenamiento in enumerate(datos):
        distancia = distanciaEuclidiana(
            punto, puntoEntrenamiento)  # calculo de la distancia

        # se agrega a distancias el valor de la distancia, la clase y el indice del punto de entrenamiento.
        distancias.append([distancia, puntoEntrenamiento[4], x+1])
    distancias.sort()
    for i in range(k):
        # se agrega la clase del punto de entrenamiento a la lista de puntos k mas cercanos
        puntosKMascercanos.append(distancias[i][1])
    claseMayoritaria = max(set(puntosKMascercanos),  # se obtiene la clase mayoritaria de los puntos k mas cercanos
                           key=puntosKMascercanos.count)

    # se agrega el punto clasificado
    puntoClasificado.append([
        punto[5], punto[0], punto[1], punto[2], punto[3], claseMayoritaria])
    return puntoClasificado


# puntos es la lista de puntos a clasificar y k es el numero de vecinos mas cercanos a considerar
def clasificarPuntos(puntos, k):
    puntosClasificados = []  # lista donde se guardan los puntos clasificados
    i = 0
    for punto in puntos:  # for que recorre los puntos a clasificar
        i += 1
        # print('Clasificando punto: ', i)
        # print("faltan" + str(len(puntos) - i))
        # Porcentaje:
        # porcentaje de puntos clasificados
        print(str((i/len(puntos))*100) + "%")
        # clasificacion del punto actual
        puntoClasificado = clasificarPunto(punto, k)
        # se agrega el punto clasificado a la lista de puntos clasificados
        puntosClasificados.append(puntoClasificado)
    return puntosClasificados


print('Clasificando puntos de prueba')
puntosClasificados = clasificarPuntos(datosPrueba, 5)

print('Puntos clasificados')
for punto in puntosClasificados:
    print(punto)
# Aqui se obtienen las clases reales y las clases predichas
clasesReales = []
clasesPredichas = []
for punto in datosPrueba:
    clasesReales.append(punto[4])

for punto in puntosClasificados:
    clasesPredichas.append(punto[0][5])


# Funcion que guarda los datos en un archivo csv
def guardarDatos(nombre_archivo, datos):
    archivo = open(nombre_archivo, 'w')
    archivo.write('timestamp, x, y, z, w, clase\n')
    for dato in datos:
        archivo.write(str(dato[0][0]) + ',' + str(dato[0][1]) + ',' + str(dato[0][2]) +
                      ',' + str(dato[0][3]) + ',' + str(dato[0][4]) + ',' + str(dato[0][5]) + '\n')
    archivo.close()


guardarDatos('participante_yh13_dataset_clasificado.csv', puntosClasificados)

# Generar matriz de confusion


def matrizConfusion(clasesReales, clasesPredichas):
    matriz = np.zeros((5, 5), dtype=int)
    for x, claseReal in enumerate(clasesReales):
        if claseReal == clasesPredichas[x]:
            if claseReal == 0:
                matriz[0][0] += 1
            elif claseReal == 1:
                matriz[1][1] += 1
            elif claseReal == 2:
                matriz[2][2] += 1
            elif claseReal == 3:
                matriz[3][3] += 1
            elif claseReal == 4:
                matriz[4][4] += 1
        else:
            if claseReal == 0:
                if clasesPredichas[x] == 1:
                    matriz[0][1] += 1
                elif clasesPredichas[x] == 2:
                    matriz[0][2] += 1
                elif clasesPredichas[x] == 3:
                    matriz[0][3] += 1
                elif clasesPredichas[x] == 4:
                    matriz[0][4] += 1
            elif claseReal == 1:
                if clasesPredichas[x] == 0:
                    matriz[1][0] += 1
                elif clasesPredichas[x] == 2:
                    matriz[1][2] += 1
                elif clasesPredichas[x] == 3:
                    matriz[1][3] += 1
                elif clasesPredichas[x] == 4:
                    matriz[1][4] += 1
            elif claseReal == 2:
                if clasesPredichas[x] == 0:
                    matriz[2][0] += 1
                elif clasesPredichas[x] == 1:
                    matriz[2][1] += 1
                elif clasesPredichas[x] == 3:
                    matriz[2][3] += 1
                elif clasesPredichas[x] == 4:
                    matriz[2][4] += 1
            elif claseReal == 3:
                if clasesPredichas[x] == 0:
                    matriz[3][0] += 1
                elif clasesPredichas[x] == 1:
                    matriz[3][1] += 1
                elif clasesPredichas[x] == 2:
                    matriz[3][2] += 1
                elif clasesPredichas[x] == 4:
                    matriz[3][4] += 1
            elif claseReal == 4:
                if clasesPredichas[x] == 0:
                    matriz[4][0] += 1
                elif clasesPredichas[x] == 1:
                    matriz[4][1] += 1
                elif clasesPredichas[x] == 2:
                    matriz[4][2] += 1
                elif clasesPredichas[x] == 3:
                    matriz[4][3] += 1
    return matriz


matriz = matrizConfusion(clasesReales, clasesPredichas)
print(matriz)

# Funcion que calcula la precision de la matriz de confusion
def precision(matriz):
    precision = 0
    for x in range(len(matriz)):
        precision += matriz[x][x]
    return precision/np.sum(matriz)


def precisionClase(matriz):
    precisionClase = []
    for x in range(len(matriz)):
        precisionClase.append(matriz[x][x]/np.sum(matriz[x]))
    return precisionClase


def recallClase(matriz):
    recallClase = []
    for x in range(len(matriz)):
        # Aqui se calcula el recall de cada clase, donde se divide la diagonal de la matriz por la suma de la columna
        recallClase.append(matriz[x][x]/np.sum(matriz[:, x]))
    return recallClase

def fscore(precisionClase, recallClase):
    fscore = []
    for x in range(len(precisionClase)):
        fscore.append(2*precisionClase[x]*recallClase[x]/(precisionClase[x]+recallClase[x]))
    return fscore

print('Precision: ', precision(matriz))
print('Precision de cada clase: ', precisionClase(matriz))
print('Promedio de precision de cada clase: ', np.mean(precisionClase(matriz)))
print('Recall de cada clase: ', recallClase(matriz))
print('Promedio de recall de cada clase: ', np.mean(recallClase(matriz)))
print('Fscore de cada clase: ', fscore(precisionClase(matriz), recallClase(matriz)))
print('Promedio de fscore de cada clase: ', np.mean(fscore(precisionClase(matriz), recallClase(matriz))))