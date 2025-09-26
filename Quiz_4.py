# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 08:45:53 2025

@author: joseA
"""
import random
import math

# Definimos las ciudades con coordenadas (x, y)
ciudades = {
    "Armenia": (0, 0),
    "Bogota": (1, 5),
    "Cali": (2, 3),
    "Duitama": (5, 2),
    "Ebejico": (6, 6)
}

def distancia_ruta(ruta):
    distancia_total = 0
    for i in range(len(ruta) - 1):
        x1, y1 = ciudades[ruta[i]]
        x2, y2 = ciudades[ruta[i+1]]
        distancia_total += math.dist((x1,y1), (x2,y2))
    # Regresamos al inicio
    x1, y1 = ciudades[ruta[-1]]
    x2, y2 = ciudades[ruta[0]]
    distancia_total += math.dist((x1,y1), (x2,y2))
    return distancia_total

def crear_poblacion_inicial(tam_poblacion):
    ciudades_lista = list(ciudades.keys())
    poblacion = []
    for _ in range(tam_poblacion):
        ruta = ciudades_lista[:]
        random.shuffle(ruta)
        poblacion.append(ruta)
    return poblacion

def seleccion(poblacion, distancias):
    torneo = random.sample(list(zip(poblacion, distancias)), 3)  # torneo con 3 rutas
    torneo.sort(key=lambda x: x[1])  # ordenamos por distancia
    return torneo[0][0]  # la mejor ruta

def cruce(padre1, padre2):
    inicio = random.randint(0, len(padre1) - 2)
    fin = random.randint(inicio, len(padre1) - 1)
    
    hijo = [None] * len(padre1)
    # copiamos segmento del padre1
    hijo[inicio:fin+1] = padre1[inicio:fin+1]
    
    # rellenamos con ciudades del padre2 en orden
    pos = (fin+1) % len(padre1)
    for ciudad in padre2:
        if ciudad not in hijo:
            hijo[pos] = ciudad
            pos = (pos+1) % len(padre1)
    return hijo

def mutacion(ruta, tasa_mutacion=0.1):
    nueva_ruta = ruta[:]
    if random.random() < tasa_mutacion:
        i, j = random.sample(range(len(ruta)), 2)
        nueva_ruta[i], nueva_ruta[j] = nueva_ruta[j], nueva_ruta[i]
    return nueva_ruta

# Creamos la población inicial
poblacion = crear_poblacion_inicial(15)

# Calculamos distancias
distancias = [distancia_ruta(r) for r in poblacion]

print("Población inicial:")
for ruta, d in zip(poblacion, distancias):
    print(ruta, "-> Distancia:", round(d, 2))
