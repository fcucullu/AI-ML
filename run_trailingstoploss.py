#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 10:29:41 2020

@author: farduh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime 


from src.Optimizer import Optimizer
from src.Function_TrailingStopLoss import get_price
from src.Function_TrailingStopLoss import Objective_Function

pairs=["BTCUSDT","ETHUSDT","BNBUSDT","LTCUSDT","ETHBTC","BNBBTC","LTCBTC"]

pair=pairs[2]

price=get_price(pair,"15m").dropna()

## defino la función objetivo a maximizar
ob_func=Objective_Function(price)
#defino el optimizador
opt=Optimizer(ob_func,
              {'a2': (0, 30),
    	       'a3': (-10, 10),
               'b2': (0, 30),
    	       'b3': (-10, 10),
    	       },
              #logs_path=True
              n_bayes_iter=400,
              fecha_inicio_train=datetime(2019,1,1),
              fecha_fin_train=datetime(2020,1,1),
              fecha_inicio_test=datetime(2020,1,1),
              fecha_fin_test=datetime(2020,3,1),
              )
df_res=opt.full_optimization()
