#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clase esquelo para definir la funciones a optimizar

"""


import pandas as pd
import numpy as np

from .ObjectiveFunctionBase import AbstractObjectiveFunction

#funcion necesaria para levantar los precios, pero el codigo en si no la necesita
def get_price( symbol, interval):

        filename = "{}_{}.csv".format(symbol, interval)
        print(filename)
        df_price=pd.read_csv("../klines/downloaded/"+filename).drop_duplicates()
        price = df_price[["open","close","low","high","close_time"]].drop_duplicates().set_index("close_time")
        price.index = pd.to_datetime(price.index)
        price.index += pd.Timedelta(1, unit="ms")
        price["close"] = pd.to_numeric(price["close"])
        price = price.sort_index(ascending=True)
        price.index=pd.DatetimeIndex([i.replace(tzinfo=None) for i in price.index])
        del df_price
        return price

#Funcion que necesita la clase Optimizador
class ObjectiveFunction(AbstractObjectiveFunction):

    def function_to_optimize(self,**kwargs): ##método necesario de la función para la optimización
        """
        Tiene de como entrada los parámetros a optimizar
        en caso de la función esperar un entero, es necesario hacerle un casting
        """
        
        sma_short=int(kwargs["short"]) 
        sma_long =int(kwargs["long"])
        
        df=pd.DataFrame()
        df["close"]=self.time_serie["close"]
        df["sma_short"]=self.time_serie["close"].rolling(sma_short).mean()
        df["sma_long"]=self.time_serie["close"].rolling(sma_long).mean()
        df["action"]=0
        df.loc[(df["sma_long"]>df["sma_short"]) ,"action"]=-1
        df.loc[(df["sma_short"]>df["sma_long"]),"action"]=1 
        df["action"]=df["action"].shift(1).replace(0,np.nan).ffill()
        df["returns"]=np.where(df["action"]==1,df["close"].pct_change(),0)
        df["returns"]=np.where(df["action"]!=df["action"].shift(1),df["returns"]-0.002,df["returns"])
        
        return (1+df["returns"].loc[self.date_init:self.date_final]).cumprod().iloc[-1]
    

