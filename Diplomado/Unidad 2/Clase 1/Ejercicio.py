# -*- coding: utf-8 -*-
"""
Created on Tue May  3 23:16:31 2022

@author: milan
"""
# Importar librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss

#Cargar CSV
titanic= pd.read_csv(r"C:\Users\milan\titanic.csv")

# Quitar nulos
#Edad=titanic['Age'].fillna(np.mean(titanic['Age']))
Edad=titanic[titanic['Age'].notna()]

#Crar Figura
fig = plt.figure(figsize=(7,7))


#Variables
mean=np.mean(Edad['Age'])
#Imprime el promedio
print("Promedio general:",mean)
std=np.std(Edad['Age'])
#Imprime STD
print("STD:",std)
#Imprime mínima y máxima
min_v=Edad['Age'].min()
print("MIN:",min_v)
max_v=Edad['Age'].max()
print("MAX:",max_v)


# Se hace la figura inicial, se le da un grosor excesivo a la linea para que esta se pueda superponer con las otras líneas que se añadiran comparando la supervivencia.
# Esto fue la mejor forma que encontré para lograr la comparación sin romper la gráfica original.
# Gráfica la campana y distribución del dataset
ax = fig.add_subplot(111)
asd = ax.hist(Edad['Age'], bins = 90 , density=True)
rv = ss.norm(mean, std)
x = np.linspace(min_v,max_v)
h = ax.plot(x, rv.pdf(x), lw=20, color="blue");

# Entrega las respuestas de las preguntas de la sección 2 de la tarea. Todas las respuestas se calculan e imprimen con las funciones de scipy.
print("Probabilidad menores de 18:",rv.cdf(18))
print("Probabilidad Mayor a 80:",rv.sf(80))
print("Probabilidadentre 20 y 40:",rv.cdf(40)-rv.cdf(20))

# A continuación se esta imprimiendo las variables resultantes de dividir el data set en 2 lotes, de supervivientes y no supervivientes.
Edad_s=Edad[(Edad.Survived == 1)]
mean_s=np.mean(Edad_s['Age'])
print("Promedio sobreviviente general:",mean_s)
std_s=np.std(Edad_s['Age'])
print("STD sobreviviente:",std_s)
min_v_s=Edad_s['Age'].min()
print("MIN sobreviviente:",min_v_s)
max_v_s=Edad_s['Age'].max()
print("MAX sobreviviente:",max_v_s)

Edad_ns=Edad[(Edad.Survived == 0)]
mean_ns=np.mean(Edad_ns['Age'])
print("Promedio no sobreviviente general:",mean_ns)
std_ns=np.std(Edad_ns['Age'])
print("STD no sobreviviente:",std_ns)
min_v_ns=Edad_ns['Age'].min()
print("MIN nosobreviviente:",min_v_ns)
max_v_ns=Edad_ns['Age'].max()
print("MAX no sobreviviente:",max_v_ns)
#  En promedio la gente más joven sobrevivio por una diferencia de 2 años, pero las camánas en la gráfica están prácticamente superpuestas.
# no diría que la edad tuvo una importancia significativa en la superviencia, yo haría responsable a otras variables en el mismo dataset.
rv_s = ss.norm(mean_s, std_s)
x_s = np.linspace(min_v_s,max_v_s)
h_s = ax.plot(x_s, rv.pdf(x_s), lw=10, color="red");

rv_ns = ss.norm(mean_s, std_s)
x_ns = np.linspace(min_v_s,max_v_s)
h_ns = ax.plot(x_ns, rv.pdf(x_ns), lw=5, color="green");


ax.set_xlabel('Edad')
ax.set_ylabel('Prob')
# mostrar gráfico
plt.show()
