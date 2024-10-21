from abc import ABC,abstractclassmethod
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class AbstractModelValidation(ABC):
    def __init__(self,target_function):
        self.target_function=target_function
        super().__init__()
    @abstractclassmethod
    def validate_models():
        pass
    def model_selection():
        pass
    
class LinearModelValidation(AbstractModelValidation):
    
    def model_selection(self,df_validation,validation_size):
        """
        Se decorara al dataframe devalidation con 4 criterios de selección para elegir la estabilidad del modelo
        
        * criteria_diff:
            diferencia absoluta entre el resultado de entrenamiento y el resultado de validación.
        * criteria_diff2:
            diferencia cuadrada entre el resultado de entrenamiento y el resultado de validación.
        * criteria_ratio:
            valor absoluto de la razón entre el resultado de entrenamiento y el resultado de validación menos 1.
        * criteria_lnratio:
            valor absoluto del logaritmo natural de la razón entre el resultado de entrenamiento y el resultado de validación.
        
        Se aplica lineal validation sobre las muestras de entrenamiento
        
        * validation_size: *float, int or None, optional (default=None)*
            If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the validation split. If int, represents the absolute number of validation samples. If None, the value is set to the complement of the train size. If train_size is also None, it will be set to 0.25.
        
        
        """
        
        ###
        
        dates=np.array(self.target_function.time_serie.index)
        index_train,index_validation=train_test_split(dates,test_size=validation_size,shuffle=False)
        
        len_train=len(index_train)
        len_validation=len(index_validation)
        
        
        df_temp=df_validation.copy()
        
        df_temp["train_result"]=(df_temp["train_result"])**(1/len_train)
        df_temp["validation_result"]=(df_temp["validation_result"])**(1/len_validation)
        
        df_temp["delta_train_result"]=df_temp["train_result"]-df_temp["train_result"].mean()
        df_temp["delta_validation_result"]=df_temp["validation_result"]-df_temp["validation_result"].mean()
        
        ###  
        
        df_validation["criteria_diff"]=0
        df_validation["criteria_diff2"]=0
        df_validation["criteria_ratio"]=1
        df_validation["criteria_lnratio"]=0
        
        
        
        ####
        
        df_validation["criteria_diff"]+=abs(df_temp["delta_train_result"]-df_temp["delta_validation_result"])
    
        df_validation["criteria_diff2"]+=(df_temp["delta_train_result"]-df_temp["delta_validation_result"])**2
    
        df_validation["criteria_ratio"]*=(df_temp["delta_train_result"]/df_temp["delta_validation_result"])
    
        df_validation["criteria_lnratio"]+=abs(np.log(df_temp["delta_train_result"]/df_temp["delta_validation_result"]))
        ####

        df_validation["criteria_ratio"]=abs(df_validation["criteria_ratio"]-1)
            
        return df_validation
    
    
    def validate_models(self,df_samples,validation_size,**kwargs):
        """
        Se aplica lineal validation sobre las muestras de entrenamiento
        
        * df_samples:
            dataframe con los hiperparámetros que van a ser validados con este método
        
        * validation_size: *float, int or None, optional (default=None)*
            If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the validation split. If int, represents the absolute number of validation samples. If None, the value is set to the complement of the train size. If train_size is also None, it will be set to 0.25.
        
        """
        dates=np.array(self.target_function.time_serie.index)
        index_train,index_validation=train_test_split(dates,test_size=validation_size,shuffle=False)
        
        dates_train={"inicial":index_train[0],
                     "final":index_train[-1]}
    
        dates_validation={"inicial":index_validation[0],
                    "final":index_validation[-1]}
            
        
        results=[]
        for i,row in df_samples.iterrows():
            dict_row=row.to_dict()
            
            self.target_function.set_dates(dates_train["inicial"],dates_train["final"])
            train_result=self.target_function.function_to_optimize(**(dict_row))
            self.target_function.set_dates(dates_validation["inicial"],dates_validation["final"])
            validation_result=self.target_function.function_to_optimize(**(dict_row))
            dict_row.update({"train_result":train_result,
                            "validation_result":validation_result})
            results.append(dict_row)
    
        return pd.DataFrame(results)
    
class WalkForwardModelValidation(AbstractModelValidation):
    
    def model_selection(self,df_validation,validation_size,n_split=4):
        """
        Se decorara al dataframe devalidation con 4 criterios de selección para elegir la estabilidad del modelo
        
        * criteria_diff:
            diferencia absoluta entre el resultado de entrenamiento y el resultado de validación.
        * criteria_diff2:
            diferencia cuadrada entre el resultado de entrenamiento y el resultado de validación.
        * criteria_ratio:
            valor absoluto de la razón entre el resultado de entrenamiento y el resultado de validación menos 1.
        * criteria_lnratio:
            valor absoluto del logaritmo natural de la razón entre el resultado de entrenamiento y el resultado de validación.
        """ 
        
        
        
        dates=np.array(self.target_function.time_serie.index)
        index_train,index_validation=train_test_split(dates,test_size=validation_size,shuffle=False)
        
        len_train=len(index_train)
        len_validation=len(index_validation)
        
        
        df_temp=df_validation.copy()
        
        
        ###
        
        df_validation["criteria_diff"]=0
        df_validation["criteria_diff2"]=0
        df_validation["criteria_ratio"]=1
        df_validation["criteria_lnratio"]=0
        
        
        ###
        for n_iter in range(0,n_split):
        
            df_temp["train_result_"+str(n_iter)]=(df_temp["train_result_"+str(n_iter)])**(1/len_train)
            df_temp["validation_result_"+str(n_iter)]=(df_temp["validation_result_"+str(n_iter)])**(1/len_validation)
        
            df_temp["delta_train_result_"+str(n_iter)]=df_temp["train_result_"+str(n_iter)]-df_temp["train_result_"+str(n_iter)].mean()
            df_temp["delta_validation_result_"+str(n_iter)]=df_temp["validation_result_"+str(n_iter)]-df_temp["validation_result_"+str(n_iter)].mean()
        
            ###
        
            df_validation["criteria_diff"]+=abs(df_temp["delta_train_result_"+str(n_iter)]-df_temp["delta_validation_result_"+str(n_iter)])
        
            df_validation["criteria_diff2"]+=(df_temp["delta_train_result_"+str(n_iter)]-df_temp["delta_validation_result_"+str(n_iter)])**2
        
            df_validation["criteria_ratio"]*=(df_temp["delta_train_result_"+str(n_iter)]/df_temp["delta_validation_result_"+str(n_iter)])
        
            df_validation["criteria_lnratio"]+=abs(np.log(df_temp["delta_train_result_"+str(n_iter)]/df_temp["delta_validation_result_"+str(n_iter)]))
        

        df_validation["criteria_ratio"]=abs(df_validation["criteria_ratio"]-1)
            
        return df_validation
    
    
    def validate_models(self,df_samples,validation_size,anclaje=False,n_split=4,**kwargs):
        """
        Se aplica walkforward sobre las muestras de entrenamiento
    
        * df_samples:
            dataframe con los hiperparámetros que van a ser validados con este método
        
        * anclaje: *False or True (default=False)*
            True en caso que se quiera hacer la validación con anclaje.
        
        * validation_size: *float, int or None, optional (default=None)*
            If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the validation split. If int, represents the absolute number of validation samples. If None, the value is set to the complement of the train size. If train_size is also None, it will be set to 0.25.
        
        """
        dates=np.array(self.target_function.time_serie.index)
        
        index_train,index_validation=train_test_split(dates,test_size=validation_size,shuffle=False)
        index_validation_array=np.array_split(index_validation,n_split)
        
        dates_train=[]
        dates_validation=[]
        train_shift=0
        
        for i in range(0,n_split):
            index_validation=index_validation_array[i]
            
            
            
            dates_train.append({"inicial":index_train[train_shift],
                                "final":dates[len(index_train)+train_shift]})
    
            dates_validation.append({"inicial":index_validation[0],
                                "final":index_validation[-1]})
            if anclaje==False:
                train_shift+=len(index_validation)
            
        
        
        
        
        results=[]
        for i,row in df_samples.iterrows():
            dict_row=row.to_dict()
            for n_iter in range(0,n_split):
                                
                self.target_function.set_dates(dates_train[n_iter]["inicial"],dates_train[n_iter]["final"])
                train_result=self.target_function.function_to_optimize(**(dict_row))
                
                self.target_function.set_dates(dates_validation[n_iter]["inicial"],dates_validation[n_iter]["final"])
                validation_result=self.target_function.function_to_optimize(**(dict_row))
                
                dict_row.update({"train_result_"+str(n_iter):train_result,
                                "validation_result_"+str(n_iter):validation_result
                                })
            results.append(dict_row)
          
            
            
        return pd.DataFrame(results)
        
    
        
