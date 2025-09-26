El código lo que hace es simular un escenario de búsqueda de sobrevivientes usando un grupo de drones coordinados con un algoritmo de optimización por enjambre de partículas (PSO). La idea es que los drones deben cubrir un área de 5 km por 5 km y cada uno tiene un sensor que detecta señales de vida dentro de un radio de 200 metros.

Primero se crea un “mapa de probabilidades” que representa en qué zonas es más probable encontrar personas, como si fueran manchas de calor. Ese mapa no es real, sino generado artificialmente usando varias funciones gaussianas para simular diferentes áreas con mayor o menor probabilidad de hallar sobrevivientes.

Luego se define cómo medir la calidad de una solución. Una solución en este caso es un conjunto de posiciones de los 10 drones. El algoritmo calcula qué parte del mapa de probabilidades queda cubierta por los radios de detección de los drones. Cuanta más probabilidad se cubra, mejor es la solución.

El PSO empieza generando muchas posibles configuraciones de drones al azar. Cada configuración se evalúa con la función de cobertura. Después, cada “partícula” del enjambre guarda la mejor solución que ha encontrado y comparte información con las demás, lo que permite que poco a poco el grupo se vaya acercando a una buena distribución. En cada paso, las posiciones de los drones se van ajustando mediante reglas que combinan tres cosas: mantener parte de su movimiento anterior (inercia), acercarse a su mejor solución personal (componente cognitiva) y acercarse a la mejor solución grupal (componente social).

Con el paso de las iteraciones, el algoritmo va mejorando y aumentando la cobertura de zonas con alta probabilidad. Al final, se obtiene un conjunto de posiciones que maximizan la probabilidad de encontrar sobrevivientes en el menor tiempo posible, al menos dentro del modelo simplificado que asume drones estáticos.

El argoritmo muestra 3 graficas:
- El mapa de probabilidades con las posiciones finales de los drones
- Un mapa que indica las zonas efectivamente cubiertas por los sensores
- Evolución del mejor resultado encontrado durante las iteraciones.
