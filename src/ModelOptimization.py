"""
more info:
    https://towardsdatascience.com/train-validation-and-test-sets-72cb40cba9e7
    https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6
    https://towardsdatascience.com/time-series-nested-cross-validation-76adba623eb9
    https://towardsdatascience.com/validation-methods-for-trading-strategy-development-1efea8284b02
    
"""
from .HyperParametersGenerator import AbstractHyperParametersGenerator
from .HyperParametersGenerator import BayesianHyperParametersGenerator

from .ModelValidation import AbstractModelValidation
from .ModelValidation import LinearModelValidation
from .ModelValidation import WalkForwardModelValidation

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

class ModelOptimization:
    def __init__(self,
                 target_function,
                 generator:AbstractHyperParametersGenerator,
                 validation:AbstractModelValidation):
        """
        * target_function: *class*
            clase dos métodos necesarios.
                1. function_to_optimize(self,**kwargs): funcion a la cual se la va a maximizar
                2. set_dates(self,fecha_inicial,fecha_final) : método para fijar fecha desde donde hasta donde se obtiene el resultado a optimizar
        * generator: *class*
            clase con la que se generaran los modelos
        * validation: *class*
            clase con la cual se validara el modelo
        
        """
        
        self.target_function=target_function
        self.generator=generator(target_function)
        self.validation=validation(target_function)
    
    
    
    def run_full_optimization(self,
                              parameter_bounds,
                              baseline_umbral,
                              test_size=0.25,
                              validation_size=0.25,
                              **kwargs):
        """
        Se realiza una optimización total del modelo
        
        * parameter_bounds: *dict*
            Límites de los parámetros a optimzar. Las variables son keys en el dictionario y los límites de las variables a optimizar son tuplas, ej:
        
        >>> pbounds = {'x': (2, 4), 'y': (-3, 3)}
        
        * baseline_umbral: *float*
            una vez generadas las muestras se realiza un filtro con un requerimiento sobre la variable a optimizar
        * test_size: *float, int  (default=0.25)*
            Si es float, debe ser un número entre 0.0 y 1.0 que representa la proporción del dataset para incluir en el test. Si es entero, representa el número absoluto de muestras de test.
            Si el valor es 0 no se considera un muestra test
        * validation_size: *float, int  (default=0.25)*
            Tamaño de la serie con la cual se validará el modelo. Si es float, debe ser un número entre 0.0 y 1.0 que representa la proporción del dataset para incluir en el test. Si es entero, representa el número absoluto de muestras de test.  
            Si el valor es 0 no se realizará ninguna validación y criterio de selección será la muestra con el resultado baseline más grande
        
        Esta función retorna una tupla con:
            * Un diccionario con el mejor resultado obtenido
            * Un pandas.DataFrame con todas las muestras generadas y su resultado
            * Un pandas.DataFrame con los resultados de las validaciones
        
        """
        
        
        dates= pd.DatetimeIndex( [date for date in self.target_function.time_serie.loc[self.target_function.date_init:self.target_function.date_final].index if (date >= self.target_function.date_init) or (date <= self.target_function.date_final)])
        
        if test_size==0:
            dates_train={"inicial":dates[0],
                         "final":dates[-1]}
        else:
            
            
            index_train,index_test=train_test_split(dates,test_size=test_size,shuffle=False)
        
        
            dates_train={"inicial":index_train[0],
                         "final":index_train[-1]}
    
            dates_test={"inicial":index_test[0],
                        "final":index_test[-1]}
        #generar muestras
        
        df_samples= self.generate_samples(parameter_bounds,dates_train["inicial"],dates_train["final"],**kwargs)

        #descarto muestras que no pasas cierto umbral
        df_samples=df_samples[df_samples["baseline"]>baseline_umbral].reset_index().drop(columns="index")
        
        if df_samples.empty:
            print("Ninguna muestra paso el umbral requerido, utilice un umbral más bajo o genere más número de muestras")
            return 0
        
        
        if validation_size==0:
            df_validation=pd.DataFrame()
            idx=df_samples["baseline"].idxmax()
        else:   
            #valido muestras
            df_validation=self.validate_models(df_samples,validation_size,**kwargs)

            #selecciono bajo algún criterio las muestras
            df_validation=self.validation.model_selection(df_validation,validation_size)
            idx=(df_samples["baseline"]**2/df_validation["criteria_diff2"]).idxmax()
        
        
        
        #obtengo el resultado del test 
        if test_size==0:
            test_result=0
        else:
            test_result=self.test_model(df_samples.loc[idx],dates_test["inicial"],dates_test["final"])
        
            #preparo diccionario con resultados
        result=df_samples.loc[idx].to_dict()
        result.update({"test_result":test_result})
        
        return result,df_samples,df_validation

    def generate_samples(self, parameter_bounds,initial_date,final_date,**kwargs):
        """
        Genera muestras con distintos hiperparametros
        * parameter_bounds: *dictionario*
            Límites de los parámetros a optimzar. Las variables son keys en el dictionario y los límites de las variables a optimizar son tuplas, ej:
        
        >>> pbounds = {'x': (2, 4), 'y': (-3, 3)}
        
        * initial_date: *datetime*
            fecha en la que empieza ejecutarse el modelo
        * initial_date: *datetime*
            fecha en la que termina ejecutarse el modelo
        * **kwargs:
            hiperparametros de los modelos generadores
        """
        self.target_function.set_dates(initial_date,final_date)
        return self.generator.generate_parameters( parameter_bounds,**kwargs)
    
    def validate_models(self,df_samples,validation_size,**kwargs):
        """
        
        * df_samples: *pandas.DataFrame*
            dataframe con los hiperparámetros que van a ser validados con este método
        
        * validation_size: *float, int*
            Tamaño de la serie con la cual se validará el modelo. Si es float, debe ser un número entre 0.0 y 1.0 que representa la proporción del dataset para incluir en la validación. Si es entero, representa el número absoluto de muestras de validación.  
            
        * **kargs:
            argumentos específicos del modelo
        """
        return self.validation.validate_models(df_samples,validation_size,**kwargs)
    
    def test_model(self,row,initial_date,final_date):
        """
        *row: *pandas.Series*
            fila con la muestra a la que se le realizará el test
        *initial_date: *datetime*
            fecha a partir en la que empieza el test
        *initial_date: *datetime*
            fecha en la que termina el test
        """
        
        result=row.to_dict()
        self.target_function.set_dates(initial_date,final_date)
        test_result=self.target_function.function_to_optimize(**result)
        
        return test_result
        
    
    @classmethod
    def defaut_bayesian_linear(cls,f):
        m=ModelOptimization(f,BayesianHyperParametersGenerator(f),LinearModelValidation(f))
        return m
    
    @classmethod
    def defaut_bayesian_walkforward(cls,f):
        m=ModelOptimization(f,BayesianHyperParametersGenerator(f),LinearModelValidation(f))
        return m

    @classmethod
    def run_defaut_bayesian_linear(cls,f):
        m=ModelOptimization(f,BayesianHyperParametersGenerator(f),LinearModelValidation(f))
        df=m.run_full_optimization()
        return df
    @classmethod
    def run_defaut_bayesian_walkforward(cls,f):
        m=ModelOptimization(f,BayesianHyperParametersGenerator(f),WalkForwardModelValidation(f))
        df=m.run_full_optimization()
        return df
     
        

