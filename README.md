

# Sobre el paquete

El paquete cuenta con los siguientes scripts dentro del mismo:

* src/AlgoritmoGenetico.py
* src/HyperParametersGenerator.py
* src/ModelOptimization.py
* src/ModelValidation.py
* src/ObjectiveFunctionBase.py



# Sobre optimización


Optimización general para modelos que contengan hiperparámetros y deban ser ajustados.

El procedimiento actual que se sigue para la optimización consta de los siguientes pasos:

0. **Definir la función a maximizar :**
   El algoritmo esta diseñado para maximizar una función escalar
   
1. **División de muestra en train y test:**
   Se divide la muestra en test y train
   
2. **HyperParametersGenerator:**
   Se generan varían muestras a través de algún algoritmo de maximización
   
3. **Umbral de aceptación:**
   Se filtra la muestra definiendo un umbral de aceptación. Esto podría ser en caso de optimizar Sharpe uno mayor a 2 o un objetivo de retorno mayor al 30 %.
   
4. **Validation:**
   Se validan las muestras generadas y se define algún criterio de selección
   
5. **Test:**
   Se evalúa el resultado con los datos test
   
## Función objetivo
   Para definir la función objetivo se necesitará definir una clase, que herede la clase **AbstractObjectiveFunction** del archivo *src/ObjectiveFunctionBase.py*
```
from src.ObjectiveFunctionBase import AbstractObjectiveFunction
class ObjetiveFunction(AbstractObjectiveFunction)
```
Esta clase se define:
```
from datetime import datetime
ObjetiveFunction(df_data,datetime(2019,1,1),datetime(2020,1,1),**kwargs)
```
donde:

* time_serie: *pandas.DataFrame*
  Contiene la serie temporal que se necesitará para usar en la función a optimizar
* date_init: *datetime*
  Fecha a partir de la que empieza la estrategia de la función
* date_final: *datetime*
  Fecha en la que termina el estrategia de la función
* \*\*kwargs: *dict*
  Contiene variables externas

dentro de esta clase se define a optimizar mediante **function_to_optimize** cuyos argumentos son sólo las variables a optimizar.
Ejemplo como definir:
```    
def function_to_optimize(self,a1,a2,a3):
```    
donde a1, a2 y a3 son las variables a optimizar.
Se puede acceder al pandas.DataFrame con la serie temporal en la función de la siguiente forma:
```
df=self.time_serie
```

Es importante que se defina el rango en el cual se esta valuando la función utilizando

```
 df_result=df_result.loc[self.date_init,self.date_final]
```
Para acceder a otra variable externas que se hayan definido en la inicialización de la clase, se debe proceder de la siguiente manera (suponiendo como variables externas "vext1" y "vext2")
```
variable_externa_1=self.variable_externa["vext1"]
variable_externa_2=self.variable_externa["vext2"]
```
## Clase ModelOptimization
        
Para inicializar la clase **ModelOptimization**, además de definir la función objetivo se necesitará selección el método para generar muestras con distintos hiperparámetros y el método de validación.
Actualmente para generar modelos con hiperparámetros distintos se encuentran los siguientes algoritmos:

* RandomHyperParametersGenerator
* GridHyperParametersGenerator
* BayesianHyperParametersGenerator
* GeneticHyperParametersGenerator

y para validarlos:

* LinearModelsValidation
* WalkForwardModelsValidation

Ejemplo de inicialización de ModelOptimization:
```
from src.HyperParametersGenerator import BayesianHyperParametersGenerator
from src.ModelValidation import WalkForwardModelValidation
from src.ModelOptimization import ModelOptimization
m=ModelOptimization(ObjFunc,BayesianHyperParametersGenerator,WalkForwardModelValidation)
```
siendo **ObjFunc** una clase con la función previa definida previamente


### Optimización completa

para correr una optimización completa se utiliza el método **run_full_optimization** que tiene como entrada:

* df_time_serie: *pandas.DataFrame*
    su índice contiene las fechas de la serie de tiempo
* parameter_bounds: *diccionario*
    Límites de los parámetros a optimizar. Las variables son keys en el diccionario y los límites de las variables a optimizar son tuplas, ej:
```
pbounds = {'x': (2, 4), 'y': (-3, 3)}
```
* baseline_umbral: *float*
    una vez generadas las muestras se realiza un filtro con un requerimiento sobre la variable a optimizar
* test_size: *float, int  (default=0.25)*
    Si es float, debe ser un número entre 0.0 y 1.0 que representa la proporción del dataset para incluir en el test. Si es entero, representa el número absoluto de muestras de test.
Si el valor es 0 no se considera un muestra test

* validation_size: *float, int  (default=0.25)*
    Tamaño de la serie con la cual se validará el modelo. Si es float, debe ser un número entre 0.0 y 1.0 que representa la proporción del dataset para incluir en el test. Si es entero, representa el número absoluto de muestras de test.
Si el valor es 0 no se realizará ninguna validación y criterio de selección será la muestra con el resultado baseline más grande

* **kwargs: *dict*
    Diccionario con argumentos específicos de los modelos de generación de hiperparámetros y de validación elegidos

Esta función retorna una tupla con:

1. Un diccionario con el mejor resultado obtenido
* Un pandas.DataFrame con todas las muestras generadas y su resultado
* Un pandas.DataFrame con los resultados de las validaciones
        
Ejemplo para realizar un optimización completa suponiendo inicializado como en el ejemplo anterior:
```
 result=m.run_full_optimization(df_time_serie,pbounds,1.2,test_size=100,validation_size=500,n_iter=100,n_split=5)
```
### Otros métodos incluidos en la clase:

Estos métodos se utilizan para en el método anterior, son necesarios en caso de querer realizar solo uno de los pasos anteriores:

#### generate_samples

Genera muestras con distintos hiperparámetros:

* parameter_bounds: *diccionario* 
  Límites de los parámetros a optimizar. Las variables son keys en el diccionario y los límites de las variables a optimizar son tuplas, ej:
```
pbounds = {'x': (2, 4), 'y': (-3, 3)}
```
* initial_date: *datetime*
    fecha en la que empieza ejecutarse el modelo
* initial_date: *datetime*
    fecha en la que termina ejecutarse el modelo
* **kwargs:
    hiperparámetros de los modelos generadores

#### validate_models(self,df_time_serie,df_samples,validation_size,**kwargs)
        
* df_time_serie: *pandas.DataFrame* 
    dataframe con el índice de fechas utilizadas para el análisis

* df_samples: *pandas.DataFrame*
    dataframe con los hiperparámetros que van a ser validados con este método

* validation_size: *float, int*
    Tamaño de la serie con la cual se validará el modelo. Si es float, debe ser un número entre 0.0 y 1.0 que representa la proporción del dataset para incluir en la validación. Si es entero, representa el número absoluto de muestras de validación.  
    
* \*\*kwargs:
    argumentos específicos del modelo
        
#### test_model(self,row,initial_date,final_date)

* row: *pandas.Series*
    serie con los hiperparámetros para realizar el test
* initial_date: *datetime*
    fecha a partir en la que empieza el test
* initial_date: *datetime*
    fecha en la que termina el test


## Búsqueda de hiperparámetros globales para la maximizar una función

### 1. Maximización Bayesian
Para utilizarla:
```
from src.HyperParametersGenerator import BayesianHyperParametersGenerator
```
Se inicializa
```
bhpg=BayesianHyperParametersGenerator(target_function_class)
```
Al momento de la maximización se utiliza el siguiente método:
```
bhpg.generate_parameters(parameter_bounds,
						n_iter=100,
						logs_path="logs/",
						save_logfile=False,
						load_logfile=False,
						save_path="bayes_opt_logs.json",
                        load_paths=["bayes_opt_logs.json"])
```

Donde:

* parameter_bounds: 
            diccionario con las variables como keys y los límites de las variables a optimizar como tuplas,por ej:
	```        
	pbounds = {'x': (2, 4), 'y': (-3, 3)}
	```
* n_iter: 
    numero de iteraciones del algoritmo
* logs_path: 
    directorio donde se guardaran los logs
* save_log: 
    save logfile
* load_log: 
    load logfiles
* save_path: 
    nombre que recibira el logfile al guardarse
* load_paths: 
    lista con nombres de los logfiles a partir de los que continuará iterando el método
          	
### 2. Maximización con Algoritmo Genético
Para utilizarla:
```
from src.HyperParametersGenerator import GeneticHyperParametersGenerator
```
Se inicializa
```
aghpg=GeneticHyperParametersGenerator(target_function_class)
```
Al momento de la maximización se utiliza el siguiente método:
```
aghpg.generate_parameters(parameter_bounds,n_population=20, pressure=4, mutation_chance=0.3, random_init_mult=1, n_iter=30)
```


* parameter_bounds: *dict*
    diccionario con las variables como keys y los límites de las variables a optimizar como tuplas,por ej:
	```
	parameter_bounds = {'x': (2, 4), 'y': (-3, 3)}
	```
* n_population: *int*
    La cantidad de individuos que habra en la poblacion
* pressure: *int*
    Cuantos individuos se seleccionan para reproduccion. Necesariamente mayor que 2
* mutation_chance: *float*
    La probabilidad de que un individuo mute. Valor entre 0 y 1        
* random_init_mult: *float*
    Multiplica el número de población aleatoria inicial para tener un muestreo más grande a partir del cual empezar
* n_iter: *int*
    Número de iteraciones

### 3. Maximización Random
Para utilizarla:
```
from src.HyperParametersGenerator import RandomHyperParametersGenerator
```
Se inicializa
```
rhpg=RandomHyperParametersGenerator(target_function_class)
```
Al momento de la maximización se utiliza el siguiente método:
```
rhpg.generate_parameters(parameter_bounds,n_samples=1000)
```
Donde:

* parameter bounds: 
    diccionario con las variables como keys y los límites de las variables a optimizar como tuplas,por ej:
	```        
	parameter_bounds = {'x': (2, 4), 'y': (-3, 3)}
	```
* n_samples: 
    numero de samples aleatorias que se desea generar
      
### 4. Maximización con Grilla

Para utilizarla:
```
from src.HyperParametersGenerator import GridHyperParametersGenerator
```
Se inicializa
```
ghpg=GridHyperParametersGenerator(target_function_class)
```
Al momento de la maximización se utiliza el siguiente método:
```
ghpg.generate_parameters(parameter_bounds,parameter_steps)
```
Donde:

* parameter_bounds: 
            diccionario con las variables como keys y los límites de las variables a optimizar como tuplas,por ej:
        ```
        pbounds = {'x':  (2, 4), 'y': (-3, 3)}
        ```
* parameter_steps: 
            diccionario con las variables como keys y los límites de las variables a optimizar como tuplas,por ej:
        ```
 	steps = {'x':  0.2, 'y': 0.3}
        ```
## Validación
Al momento de validar los resultados baseline se puede utilizar los métodos de validación nombrados a continuación

### 1. Validación lineal
Para utilizarla:
```
from src.ModelValidation import LinearModelValidation
```
Se inicializa
```
lmv=LinearModelValidation(target_function_class)
```
Al momento de la maximización se utiliza el siguiente método:
```
lmv.validate_models(df_samples,validation_size)
```
Donde:

* df_samples:
    dataframe con los hiperparámetros que van a ser validados con este método

* validation_size: *float, int or None, optional (default=None)*
    If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the validation split. If int, represents the absolute number of validation samples. If None, the value is set to the complement of the train size. If train_size is also None, it will be set to 0.25.
        
### 2. Validación Walk-Forward

Para utilizarla:
```
from src.ModelValidation import WalkForwardModelValidation
```
Se inicializa
```
wfmv=LinearModelValidation(target_function_class)
```
Al momento de la maximización se utiliza el siguiente método:
```
wfmv.validate_models(df_samples,validation_size,anclaje=False,n_split=4)
```
Donde:

* df_samples:
    dataframe con los hiperparámetros que van a ser validados con este método

* validation_size: *float, int or None, optional (default=None)*
    If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the validation split. If int, represents the absolute number of validation samples. If None, the value is set to the complement of the train size. If train_size is also None, it will be set to 0.25.
       
 * anclaje: *False or True (default=False)*
            True en caso que se quiera hacer la validación con anclaje.
 * n_split: int
 	       Define el número de divisiones que utilizará el walkforward method



