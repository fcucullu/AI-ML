#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ejemplo
"""
from datetime import datetime
from src.Optimizer import Optimizer
from src.Function_Example import get_price
from src.Function_Example import ObjectiveFunction
from src.HyperParametersGenerator import BayesianHyperParametersGenerator
from src.ModelValidation import WalkForwardModelValidation
from src.ModelOptimization import ModelOptimization

##defino DataFrame con precios
btc_price=get_price("BTCUSDT","2h")
## defino la función objetivo a maximizar
ob_func=ObjectiveFunction(btc_price,datetime(2019,1,1),datetime(2020,1,1))
#defino el optimizador
opt=ModelOptimization(ob_func,
                      BayesianHyperParametersGenerator,
                      WalkForwardModelValidation
              )
#aplico este método que me devuelve un DataFrame con los resultados test y train para un conjunto de parámetros
result=opt.run_full_optimization(
        {"short":(1,30),"long":(2,100)},
        baseline_umbral=1.5,
        test_size=0.2,
        validation_size=0.2,
        n_iter=100,
        n_split=5
        )

print(result)