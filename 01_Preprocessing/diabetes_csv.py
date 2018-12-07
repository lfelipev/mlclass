#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Atividade para trabalhar o pré-processamento dos dados.

Criação de modelo preditivo para diabetes e envio para verificação de peformance
no servidor.

@author: Aydano Machado <aydano.machado@gmail.com>
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import requests

print('\n - Lendo o arquivo com o dataset sobre diabetes')
data = pd.read_csv('diabetes_dataset.csv')

#%%


# main libraries
import pandas as pd
import numpy as np
import time

# visual libraries
from matplotlib import pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D 
plt.style.use('ggplot')

# sklearn libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef,classification_report,roc_curve
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer
import missingno as msno
import matplotlib.gridspec as gridspec


df = pd.read_csv('diabetes_dataset.csv')
df.head()
print(df.shape)
#print(df.shape)


msno.matrix(df)
#%%

df.isnull().sum()

#%%

### IMPUTACAO
imputer_glucose = Imputer(missing_values=np.nan, strategy='mean', axis=0)
imputer_bloodpressure = Imputer(missing_values=np.nan, strategy='mean', axis=0)
imputer_skinthickness = Imputer(missing_values=np.nan, strategy='median', axis=0)
imputer_insulin = Imputer(missing_values=np.nan, strategy='median', axis=0)
imputer_bmi = Imputer(missing_values=np.nan, strategy='mean', axis=0)

df[['Glucose']] = imputer_glucose.fit_transform(df[['Glucose']])
df[['BloodPressure']] = imputer_bloodpressure.fit_transform(df[['BloodPressure']])
df[['SkinThickness']] = imputer_skinthickness.fit_transform(df[['SkinThickness']])
df[['Insulin']] = imputer_insulin.fit_transform(df[['Insulin']])
df[['BMI']] = imputer_bmi.fit_transform(df[['BMI']])


### PESOS
w_preg = 0.05
w_gluc = 2.0
w_bp = 1.06
w_insulin = 2.00
w_st = 0.05
w_bmi = 1.0
w_dpf = 0.89
w_age = 0.01


## NORMALIZACAO

def norm(dataset):
    dataNorm=((dataset-dataset.min())/(dataset.max()-dataset.min()))
    return dataNorm

# NORMALIZA
df = norm(df)
#%%


## APLICA OS PESO
df['Pregnancies'] = df['Pregnancies'] * w_preg
df['Glucose'] = df['Glucose'] * w_gluc
df['BloodPressure'] = df['BloodPressure'] * w_bp
df['SkinThickness'] = df['SkinThickness'] * w_st
df['Insulin'] = df['Insulin'] * w_insulin
df['BMI'] = df['BMI'] * w_bmi
df['DiabetesPedigreeFunction'] = df['DiabetesPedigreeFunction'] * w_dpf
df['Age'] = df['Age'] * w_age

df.isnull().sum()

#%%

data = df

#%%

# Criando X and y par ao algorítmo de aprendizagem de máquina.\
print(' - Criando X e y para o algoritmo de aprendizagem a partir do arquivo diabetes_dataset')
# Caso queira modificar as colunas consideradas basta algera o array a seguir.
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = data[feature_cols]
y = data.Outcome

# Ciando o modelo preditivo para a base trabalhada
print(' - Criando modelo preditivo')
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)

#realizando previsões com o arquivo de
print(' - Aplicando modelo e enviando para o servidor')
data_app = pd.read_csv('diabetes_app.csv')
#%%

#data_app = ((data_app-data_app.min())/(data_app.max()-data_app.min()))*20
data_app = norm(data_app)

df['Pregnancies'] = df['Pregnancies'] * w_preg
df['Glucose'] = df['Glucose'] * w_gluc
df['BloodPressure'] = df['BloodPressure'] * w_bp
df['SkinThickness'] = df['SkinThickness'] * w_st
df['Insulin'] = df['Insulin'] * w_insulin
df['BMI'] = df['BMI'] * w_bmi
df['DiabetesPedigreeFunction'] = df['DiabetesPedigreeFunction'] * w_dpf
df['Age'] = df['Age'] * w_age

#%%


y_pred = neigh.predict(data_app)

# Enviando previsões realizadas com o modelo para o servidor
URL = "https://aydanomachado.com/mlclass/01_Preprocessing.php"

#TODO Substituir pela sua chave aqui
DEV_KEY = "Chapa Participação e Transparência"

# json para ser enviado para o servidor
data = {'dev_key':DEV_KEY,
        'predictions':pd.Series(y_pred).to_json(orient='values')}

# Enviando requisição e salvando o objeto resposta
r = requests.post(url = URL, data = data)

# Extraindo e imprimindo o texto da resposta
pastebin_url = r.text
print(" - Resposta do servidor:\n", r.text, "\n")