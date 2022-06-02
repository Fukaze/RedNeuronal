import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

celsius = np.array([-40, -10, 0, 8, 15, 22, 38],dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100],dtype=float)

#capa = tf.keras.layers.Dense(units = 1, input_shape=[1])  #Dense: capa que conecta todas sus neuronas con las de la capa siguiente / units nro de neuronas / input_shape define las neuronas de entrada     /  ESTE ESQUEMA AHORRA LA CAPA DE ENTRADA
#modelo = tf .keras.Sequential([capa])  # modelo secuencial es un modelo básico

oculta1 = tf.keras.layers.Dense(units = 3, input_shape=[1])
oculta2 = tf.keras.layers.Dense(units = 3)                    #segunda capa no ocupa input_shape pq no simplifica la ENTRADA
salida =  tf.keras.layers.Dense(units = 1)

modelo = tf.keras.Sequential([oculta1,oculta2,salida])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

print("Comenzando entrenamiento...")
historial = modelo.fit(celsius,fahrenheit, epochs=1000, verbose=False)
print("modelo entrenado!")

#No sirve
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])
plt.show()

print("Predicción:")
resultado = modelo.predict([100])
print("El resultado es" + str(resultado) + " fahrenheit")

print("Variables internas del modelo")
print(oculta1.get_weights())
print(oculta2.get_weights())
print(salida.get_weights())
