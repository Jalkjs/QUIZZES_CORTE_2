# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 08:37:26 2025

@author: joseA
"""
# NOTA: Por simplicidad cada drone permanece en una posición (hover) y se evalúa
# la cobertura por unión de radios. Esto es una aproximación útil como primer modelo.
# Ajustes (velocidad, rutas múltiples y restricciones de tiempo) pueden añadirse
# en iteraciones futuras.

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Parámetros del problema
AREA_SIZE = 5000.0 # metros
GRID_N = 100  # resolución del grid para evaluar probabilidades
DRONES = 10
SENSOR_R = 200.0  # metros
SWARM_SIZE = 50
ITERATIONS = 200

xs = np.linspace(0, AREA_SIZE, GRID_N)
ys = np.linspace(0, AREA_SIZE, GRID_N)
xx, yy = np.meshgrid(xs, ys)
grid_points = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (GRID_N^2, 2)

#MAPA DE PROBABILIDADES> SIMULA ZONAS CON MAYOR PROBABILIDAD
def gaussian_blob(cx, cy, sigma, amplitude=1.0):
    d2 = (xx - cx)**2 + (yy - cy)**2
    return amplitude * np.exp(-d2 / (2 * sigma**2))

#DEFINICION DE ALGUNOS CENTROS ALEATORIOS DE ALTA PROBABILIDAD

n_blobs = 6
blobs = []
for i in range(n_blobs):
    cx = np.random.uniform(0.1*AREA_SIZE, 0.9*AREA_SIZE)
    cy = np.random.uniform(0.1*AREA_SIZE, 0.9*AREA_SIZE)
    sigma = np.random.uniform(200.0, 800.0)
    amp = np.random.uniform(0.6, 1.2)
    blobs.append((cx, cy, sigma, amp))

prob_map = np.zeros_like(xx)
for (cx, cy, sigma, amp) in blobs:
    prob_map += gaussian_blob(cx, cy, sigma, amplitude=amp)


prob_map = prob_map / prob_map.max()
prob_vec = prob_map.ravel()  # vector de probabilidades por celda

#FUNCION OBJETIVO: UNA MATRIZ DE POSICIONES DE DRONES (DRONES X 2) Y DEVUELVE COBERTURA TOTAL

def coverage_of_positions(positions):
    
    dp = grid_points[:, None, :] - positions[None, :, :]  # (n_cells, DRONES, 2)
    dists = np.linalg.norm(dp, axis=2)
    detected = (dists <= SENSOR_R)
    detected_any = detected.any(axis=1)
    covered_prob = (prob_vec * detected_any).sum()
    return covered_prob

#PSO BASICO (MINIMIZA EL COVERAGE PARA TRANSFORMAR A PROBLEMA DE MINIMIZACION
dim = DRONES * 2
bounds_min = np.zeros(dim)
bounds_max = np.ones(dim) * AREA_SIZE

#INICIO DE EMJABRE
X = np.random.uniform(bounds_min, bounds_max, size=(SWARM_SIZE, dim))  # posiciones
V = np.zeros_like(X)
pbest = X.copy()
pbest_val = np.array([-coverage_of_positions(p.reshape(DRONES,2)) for p in pbest])
gbest_idx = pbest_val.argmin()
gbest = pbest[gbest_idx].copy()
gbest_val = pbest_val[gbest_idx]

#HIPERPARAMETRO PSO
w = 0.7  # inercia
c1 = 1.4  # cognitivo
c2 = 1.4  # social

history_best = []

for it in range(ITERATIONS):
    fitness = np.array([-coverage_of_positions(p.reshape(DRONES,2)) for p in X])
    improved = fitness < pbest_val
    pbest[improved] = X[improved].copy()
    pbest_val[improved] = fitness[improved]
    idx = pbest_val.argmin()
    if pbest_val[idx] < gbest_val:
        gbest_val = pbest_val[idx]
        gbest = pbest[idx].copy()
    history_best.append(-gbest_val)
    
    
    r1 = np.random.rand(SWARM_SIZE, dim)
    r2 = np.random.rand(SWARM_SIZE, dim)
    V = w*V + c1*r1*(pbest - X) + c2*r2*(gbest - X)
    X = X + V
    
    X = np.clip(X, bounds_min, bounds_max)

# RESULTADO, MEJOR SOLUCION

best_positions = gbest.reshape(DRONES, 2)
best_covered = coverage_of_positions(best_positions)

#CALCULA MAPA DE COBERTURA

dp_final = grid_points[:, None, :] - best_positions[None, :, :]
dists_final = np.linalg.norm(dp_final, axis=2)
detected_final = (dists_final <= SENSOR_R).any(axis=1)
coverage_map = detected_final.reshape(GRID_N, GRID_N).astype(float)


plt.figure(figsize=(10, 9))
plt.subplot(2,2,1)
plt.title("Mapa de probabilidades (normalizado)")
plt.imshow(prob_map, origin='lower', extent=[0, AREA_SIZE, 0, AREA_SIZE], aspect='auto')
plt.scatter(best_positions[:,0], best_positions[:,1], marker='x')
plt.xlabel("m")
plt.ylabel("m")

plt.subplot(2,2,2)
plt.title("Mapa de cobertura por drones (1=cubierto)")
plt.imshow(coverage_map, origin='lower', extent=[0, AREA_SIZE, 0, AREA_SIZE], aspect='auto')
plt.scatter(best_positions[:,0], best_positions[:,1], marker='x')
plt.xlabel("m")

plt.subplot(2,1,2)
plt.title("Evolución de la cobertura (mejor encontrada)")
plt.plot(history_best)
plt.xlabel("Iteración")
plt.ylabel("Probabilidad cubierta total (suma)")

plt.tight_layout()
plt.show()

print(f"Probabilidad total cubierta (suma de probabilidades en celdas cubiertas): {best_covered:.4f}")
print(f"Porcentaje de área (celdas) cubiertas: {coverage_map.mean()*100:.2f}%")
print("Posiciones finales de los drones (x, y) en metros:")
for i, (x, y) in enumerate(best_positions):
    print(f" Drone {i+1:02d}: ({x:.1f}, {y:.1f})")
