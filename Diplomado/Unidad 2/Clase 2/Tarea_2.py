# -*- coding: utf-8 -*-
"""
Created on Mon May 16 16:46:11 2022

@author: Gamer
"""

import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from itertools import combinations, chain

#https://cnpubf.com/excel-file-format-cannot-be-determined-you-must-specify-an-engine-manually/
crime = pd.read_excel('crime_usa.xls', engine='xlrd')
variables = list(crime.columns)
variables_indep = variables[1:]

#Usando internet
#https://stackoverflow.com/questions/67992739/how-to-test-all-possible-iterations-in-a-multiple-linear-regresion-and-return-th
def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

modelos = pd.DataFrame({'Modelos':[], 'R^2_adj':[]})
for subset in powerset(crime.columns[1:]):
    if len(subset) > 0:
        model = sm.OLS(crime['X1'], sm.add_constant(crime[list(subset)])).fit()
        aux = pd.DataFrame({'Modelos':[subset], 'R^2_adj':[model.rsquared_adj]})
        modelos = pd.concat([modelos, aux], ignore_index=True)
        



#Una forma más "a mano"
variables = list(crime.columns)
variables_indep = variables[1:]

combinaciones = []
for j in range(len(variables_indep)+1):
    a = combinations(variables_indep, j)
    for i in a:
        aux = list(i)
        combinaciones.append(aux)
  
modelos = pd.DataFrame({'Modelos':[], 'R^2_adj':[], 'AIC':[]})
for combinacion in combinaciones:
    if len(combinacion) >= 0:
        model = sm.OLS(crime['X1'], sm.add_constant(crime[list(combinacion)])).fit()
        aux = pd.DataFrame({'Modelos':[combinacion], 'R^2_adj':[model.rsquared_adj], 'AIC':[model.aic]})
        modelos = pd.concat([modelos, aux], ignore_index=True)