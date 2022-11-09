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


def leerDatosPrueba(nombre_archivo):
    archivo = open(nombre_archivo, 'r')
    lineas = archivo.readlines()
    archivo.close()
    objetos = np.zeros((len(lineas)-1, 5))

    for x, linea in enumerate(lineas):
        if x > 0:  # se salta la primera linea
            #linea = linea.replace('\n', '')
            atributos = linea.split(',')
            objetos[x-1][0] = atributos[1]
            objetos[x-1][1] = atributos[2]
            objetos[x-1][2] = atributos[3]
            objetos[x-1][3] = atributos[4]
            objetos[x-1][4] = atributos[5]
        else:
            print('Se salta la primera linea')
            print(linea)
    return objetos


datosPrueba = leerDatosPrueba('participante_yh13_dataset.csv')

print(datosPrueba)
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

# Funcion que recorre los datos de prueba y los suma a los datos de entrenamiento uno a uno


def clasificar(datos_entrenamiento, datos_prueba, k):
    # # Suma de los datos de prueba:
    sumaDatosDePrueba = np.sum(datos_prueba, axis=0)
    # sumaDatosDePrueba = 0
    # for datoPrueba in datos_prueba:
    #     sumaDatosDePrueba += datoPrueba
    print('Suma de los datos de prueba: ' + repr(sumaDatosDePrueba))
    matriz = np.zeros((len(datos_entrenamiento), len(datos_entrenamiento[0])))

    for x, datoEntrenamiento in enumerate(datos_entrenamiento):

        matriz[x] = datoEntrenamiento + sumaDatosDePrueba

    return matriz


suma = clasificar(entrenamiento, prueba, 3)

print('Suma de los datos de prueba y entrenamiento: ')
print(suma)


matriz = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

matriz2 = np.array([[4, 5, 6],
                    [7, 8, 9],
                    [2, 3, 4],
                    [7, 8, 9],
                    [2, 3, 4]])

suma = clasificar(matriz, matriz2, 3)

print("Suma de matrices: ")
print(suma)
