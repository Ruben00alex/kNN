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
import timeit
import numpy as np

# Leer datos de entrenamiento


def leerDatosEntrenamiento(nombre_archivo):
    archivo = open(nombre_archivo, 'r')  # abrimos el archivo
    lineas = archivo.readlines()  # separamos las lineas del archivo
    archivo.close()
    # matriz de numpy
    objetos = np.zeros((len(lineas), len(lineas[0].split(','))))
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
    return objetos


datos = leerDatosEntrenamiento('participate_y16_training.csv')

# print(datos)

# Funcion que separa datos en entrenamiento y prueba (30% entrenamiento, 70% prueba) con muestras únicas y aleatorias


def separarDatos(datos):
    # aqui se mezclan los datos para que no se repitan
    np.random.shuffle(datos)
    # el slice es de 0 a 70% de los datos
    testing = datos[:int(len(datos) * 0.7)]
    # el slice es de 70% a 100% de los datos
    training = datos[int(len(datos) * 0.7):]
    return training, testing


entrenamiento, prueba = separarDatos(datos)

# Numero total de objetos en el conjunto de datos
print('Numero total de objetos en el conjunto de datos: ' + repr(len(datos)))

# Numero de objetos en el conjunto de entrenamiento
print('Numero de objetos en el conjunto de entrenamiento: ' +
      repr(len(entrenamiento)))

# Numero de objetos en el conjunto de prueba
print('Numero de objetos en el conjunto de prueba: ' + repr(len(prueba)))

# Porcentaje de objetos en el conjunto de entrenamiento
print('Porcentaje de objetos en el conjunto de entrenamiento: ' +
      repr(len(entrenamiento)/len(datos)))

# Porcentaje de objetos en el conjunto de prueba
print('Porcentaje de objetos en el conjunto de prueba: ' +
      repr(len(prueba)/len(datos)))

print('Datos de entrenamiento: ')
print(entrenamiento)

print('Datos de prueba: ')
print(prueba)

# Funcion que convierte la matriz de 4 columnas (X , Y , Y , Z) a 3 columnas (X , Y , Z), tomando el promedio de los valores de Y


def convertirMatriz(matriz):
    # crear matriz de 3 columnas
    matriz_convertida = np.zeros((len(matriz), 3))
    # recorrer matriz
    for i in range(len(matriz)):
        # asignar valores a la matriz nueva
        matriz_convertida[i][0] = matriz[i][0]
        matriz_convertida[i][1] = (matriz[i][1] + matriz[i][2]) / 2
        matriz_convertida[i][2] = matriz[i][3]
    return matriz_convertida

# Funcion que calcula la distancia euclidiana entre dos puntos(x,y,z)


def distanciaEuclidiana(x, y):
    # distancia = raiz((x1-x2)^2+(y1-y2)^2+(z1-z2)^2)
    distancia = np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2 + (x[2] - y[2])**2)
    return distancia


pruebaConvertidaAPuntosXYZ = convertirMatriz(prueba)
print('Datos de prueba convertidos a puntos XYZ: ')
print(pruebaConvertidaAPuntosXYZ)


def clasificarKNN(k, punto):
    # Lista de distancias
    distancias = []

    for dato in pruebaConvertidaAPuntosXYZ:
        # calcular la distancia entre el punto y el dato
        distancia = distanciaEuclidiana(punto, dato)
        # agregar la distancia a la lista de distancias
        distancias.append(distancia)
    # ordenar la lista de distancias
    distancias.sort()
    print('Distancias: ')
    print(distancias)
    # obtener los k vecinos mas cercanos
    vecinos = distancias[:k]
    # obtener la clase de los vecinos mas cercanos
    clases = []
    # obtener la clase mas común
    clase = max(set(clases), key=clases.count)
    return clase


# Funcion que recorre los datos de entrenamiento y los clasifica con la funcion clasificarKNN
def clasificarDatos(k):
    entrenamientoConvertidoAPuntosXYZ = convertirMatriz(entrenamiento)
    # Lista de clases
    clases = []
    # Recorrer datos de prueba
    for dato in entrenamientoConvertidoAPuntosXYZ:
        # clasificar dato
        clase = clasificarKNN(k, dato)
        print('Clase: ' + repr(clase))
        # agregar clase a la lista de clases
        clases.append(clase)
    return clases


print("Clasificacion de datos: ")
print(clasificarDatos(3))


# def clasificar(datos_entrenamiento, datos_prueba, k):
#     # inicializar la matriz de predicciones
#     predicciones = np.zeros(len(datos_prueba))
#     # recorrer los datos de prueba
#     for i in range(len(datos_prueba)):
#         # inicializar la matriz de distancias
#         distancias = np.zeros(len(datos_entrenamiento))
#         # recorrer los datos de entrenamiento
#         for j in range(len(datos_entrenamiento)):
#             # calcular la distancia euclidiana entre el punto de prueba y el punto de entrenamiento
#             distancias[j] = distancia_euclidiana(
#                 datos_prueba[i], datos_entrenamiento[j])
#         # ordenar las distancias de menor a mayor
#         indices = np.argsort(distancias)
#         # inicializar la matriz de vecinos
#         vecinos = np.zeros(k)
#         # recorrer los vecinos
#         for j in range(k):
#             # almacenar el vecino en la matriz de vecinos
#             vecinos[j] = datos_entrenamiento[indices[j]][-1]
#         # obtener la clase mas comun
#         # bincount cuenta las ocurrencias de cada valor en un array de enteros, y argmax devuelve el indice del valor maximo
#         predicciones[i] = np.argmax(np.bincount(vecinos))
#     return predicciones


#suma = clasificar(entrenamiento, prueba, 3)

#print('Suma de los datos de prueba y entrenamiento: ')
# print(suma)


# matriz = np.array([[1, 2, 3],
#                    [4, 5, 6],
#                    [7, 8, 9]])

# matriz2 = np.array([[4, 5, 6],
#                     [7, 8, 9],
#                     [2, 3, 4],
#                     [7, 8, 9],
#                     [2, 3, 4]])

# suma = clasificar(matriz, matriz2, 3)

# print("Suma de matrices: ")
# print(suma)
