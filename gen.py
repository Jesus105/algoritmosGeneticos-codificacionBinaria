
import math
import numpy as np
import matplotlib.pyplot as plt
import random
from operator import itemgetter
import matplotlib.pyplot as plt

def fun_esfera(x):
  sum=0
  n=0
  while n<5:
    sum=sum+x[n]
    n=n+1
  return sum

def fun_rosenbrock(x):
  sum=0
  n=0
  while n<4:
    sum=sum + 100*((x[n]**2)-x[n+1])**2 + (1-x[n])**2
    n=n+1
    
  return sum

def fun_ackley(x):
  sum=0
  sumCos=0
  n=0
  while n<5:
    sum=sum + x[n]**2
    sumCos=sumCos + math.cos(2*math.pi*x[n])
    n=n+1
  
  res= -20 * math.exp(-0.2*math.sqrt(0.1*sum))- math.exp(0.1*sumCos) + 20 + math.e
  return res
    

def inicializarPoblacionBinaria(tam,numVar):
  population = []
  for i in range(tam):
    individual = []
    for j in range(numVar):
      # Crear una cadena binaria aleatoria
      binary_string = ''.join(random.choices(['0', '1'], k=4))
      individual.append(binary_string)
    
    population.append(individual)    
  return population
    
def redondear(num):
    return round(num,2)

# Función de decodificación
def decode(individual):
    x = []
    for i in range(nVariables):
        x_dec = int(individual[i], 2)
        #Reparación y redondeo
        x_scaled = limInf + (limSup - limInf) * x_dec / (2 ** 4 - 1)
        x_scaled=redondear(x_scaled)
        x.append(x_scaled)
        
    return x

def fitnesss(poblacion,tipo):
    fitness=[]
    for individuo in poblacion:
        x=  (individuo,tipo)
        fitness.append(x)
    return fitness
        

def evaluar(individual, tipo):
  x = decode(individual)
  if tipo==0:
    X=fun_rosenbrock(x)
  elif tipo==1:
    X=fun_ackley(x)
  elif tipo==2:
    X=fun_esfera(x)
  
  return X


# Función de selección por torneo binario 
def torneoBinario(fitness_values):
    indices=[]
    i=0
    while i<(tamPoblacion**2):
        # Seleccionar dos individuos aleatorios
        i1 = random.randint(0, tamPoblacion- 1)
        i2 = random.randint(0, tamPoblacion - 1)
        while i2==i1:
            i2 = random.randint(0, tamPoblacion - 1)
        
        if fitness_values[i1] <= fitness_values[i2]:
            indices.append(i1)
        else:
            indices.append(i2)
        i=i+1
    return indices
  
# Función de selección por ruleta
def seleccion_ruleta(poblacion, fitness):
    # Normalizar el fitness
    probabilidad = fitness / np.sum(fitness)
    probAcumulada = np.cumsum(probabilidad)
    r = np.random.uniform(0, 1, size=len(poblacion))
    # Seleccionar los padres
    padres = []
    for i in range(len(poblacion)):
        index = np.where(probAcumulada >= r[i])[0][0]
        padres.append(poblacion[index])
        
    return padres

# Función de cruza de 2 puntos  // binaria
def cruza2puntos(parent1, parent2):
    # Seleccionar dos puntos de corte aleatorios
    cut_point1 = random.randint(0, nVariables - 2)
    cut_point2 = random.randint(cut_point1 + 1, nVariables - 1)
    child1 = parent1[:cut_point1] + parent2[cut_point1:cut_point2] + parent1[cut_point2:]
    child2 = parent2[:cut_point1] + parent1[cut_point1:cut_point2] + parent2[cut_point2:]
    return child1, child2

#cruza uniforme
def cruzaUniforme(individuo1, individuo2, indpb=0.5):
    size = min(len(individuo1), len(individuo2))
    for i in range(size):
        if random.random() < indpb:
            individuo1[i], individuo2[i] = individuo2[i], individuo1[i]
    return individuo1, individuo2

#crea poblacion de descendientes
def creaHijos(pCruza, indices, padres, tamPob):
  hijos = []
  j = 0
  while j < tamPob:
    if random.uniform(0,1) <= pCruza:
      p1 = padres[indices[j]]
      p2 = padres[indices[j+1]]
      h1, h2 = cruza2puntos(p1, p2)
      hijos.append(h1)
      hijos.append(h2)
    j += 2
  return hijos

def creaHijosR(pCruza, indices, tamPob, tipoCruza):
  hijos = []
  j = 0
  while j < tamPob:
    if random.uniform(0,1) <= pCruza:
      p1 = indices[j]
      p2 = indices[j+1]
      if tipoCruza==0:
        h1, h2 = cruza2puntos(p1, p2)
      elif tipoCruza==1:
        h1, h2 = cruzaUniforme(p1, p2)
      hijos.append(h1)
      hijos.append(h2)
    j += 2
  return hijos
  
# Función de mutación
def mutar(individuo,porcMuta):
  if random.uniform(0,1) <= porcMuta:
    # Seleccionar un bit aleatorio para mutar
    i = random.randint(0, nVariables - 1)
    j = random.randint(0, 3)
    # Cambiar el bit
    if individuo[i][j] == '0':
        individuo[i] = individuo[i][:j] + '1' + individuo[i][j+1:]
    else:
        individuo[i] = individuo[i][:j] + '0' + individuo[i][j+1:]
    
  return individuo


def algoritmoGenetico(nVariables, limInf, limSup, tamPoblacion, porcCruza, porcMuta,tipo, tipoCruza):
  padres = inicializarPoblacionBinaria(tamPoblacion,nVariables)
  print("Poblacion inicial ", padres)
  print("===========================")
  mFitness=fitnesss(padres,tipo)
  seleccion_ruleta(padres, mFitness)
  mFitness.sort()
  mejorFitness=mFitness[0]
  peorFitness=mFitness[len(padres)-1]
  fitnessEsperado=0.0000
  mejores = []
  peores = []
  promedio = []
  #repite mientras no se alcance criterio de paro
  gen=0
  diferenciaMejor=mejorFitness-fitnessEsperado
  criterioParoDiferencia= peorFitness-mejorFitness
  while ( (abs(diferenciaMejor)>0.001) & (abs(criterioParoDiferencia)>0.001) ): 
        gen=gen+1
        print("--------------------------------")
        print("No. de Generacion=")
        print(gen)
        fitnessV= fitnesss(padres,tipo)
        indices= seleccion_ruleta(padres, fitnessV)
        print("Parejas", indices)
        
        #Cruza - Generecion de hijos
        hijos = creaHijosR(porcCruza, indices, tamPoblacion,tipoCruza)

        #Mutacion
        hijos2 = []
        for hijo in hijos:
            mutado= mutar(hijo,porcMuta)
            hijos2.append(mutado)
    
        #Nueva poblacion 
        nuevaPoblacion = padres + hijos2
        print("longitud de padres e hijos", len(nuevaPoblacion))
        print("nueva poblacion", nuevaPoblacion)
    
        aSobrevivir=[]
        for individual in nu  evaPoblacion:
            fx=evaluar(individual,tipo)
            aSobrevivir.append([individual,fx])
  
        #selecciona los sobrevivientes
        sobrevivientes = sorted(aSobrevivir, key=itemgetter(1))    
        print("ordenada", sobrevivientes)
        padres = sobrevivientes[0: tamPoblacion]
        print("nuevos padres (sobrevivientes)", padres)
        
        #registra los valores del mejor y peor individuo por generación
        mej=redondear(padres[0][1])
        mejores.append(mej)
        peo=redondear(padres[-1][1])
        peores.append(peo)
        #calcula la aptitud promedio de la población en cada generación
        prom = 0
        for p in padres:
            prom += p[1]
        pro=redondear(prom/len(padres))
        promedio.append(pro)
        
        mejorFitness=padres[0][1]
        peorFitness=padres[-1][1]
        Npadres=[]
        for p in padres:
            Npadres.append(p[0])
        padres=Npadres
        
        diferenciaMejor=mejorFitness-fitnessEsperado
        criterioParoDiferencia= peorFitness-mejorFitness
                
  mFitness=fitnesss(padres, tipo)
  print("/////////////////")
  print("No. Generaciones Totales=", gen)
  print("mejor solucion - Individuo", padres[0])
  print("mejor solucion - Aptitud", mFitness[0])
  print("Criterio de paro:")
  if criterioParoDiferencia <= 0.001:
    print("Se detuvo por diferencia minima entre el mejor y el peor")
  elif diferenciaMejor<=0.001:
    print("Se detuvo por diferencia minima entre el mejor y el valor obtimo esperado")

  return mejores, peores, promedio, gen


  
def graficas(mejores, peores, promedio, generaciones, punto_size=5, punto_style='o', linea_style='-'):
    if len(mejores) != len(peores) or len(mejores) != len(promedio):
        print("Error: las listas deben tener la misma longitud")
        return

    x = list(range(1, generaciones+1))

    plt.plot(x, peores, color='red', label='peor', linestyle=linea_style, marker=punto_style, markersize=punto_size)
    plt.plot(x, promedio, color='blue', label='promedio', linestyle=linea_style, marker=punto_style, markersize=punto_size)
    plt.plot(x, mejores, color='green', label='mejor', linestyle=linea_style, marker=punto_style, markersize=punto_size)
    
    plt.legend()
    plt.xlabel('Generaciones')
    plt.ylabel('Aptitud')
    plt.title("Gráfica de convergencia")

    plt.show()
  
  
  
#parametros de entrada
nVariables = 5

       #esfera   #ros     #ackley
       #-5.12     #-2.05   #-32.77
       #5.12      #2.05    #32.77
limInf=-5.12 
limSup=5.12 

# el tamaño de la población debe ser un numero par
tamPoblacion = 20
porcCruza = 0.5
porcMuta = 0.01

#0 = ros   # 1=  ackley   #2=esfera
tipo=2
#0= cruza de dos puntos   #1=cruza uniforme
tipoCruza=1


mejores, peores, promedio, gen = algoritmoGenetico(nVariables, limInf, limSup, tamPoblacion, porcCruza, porcMuta, tipo, tipoCruza)
graficas(mejores, peores, promedio, gen)


#random.seed(123)