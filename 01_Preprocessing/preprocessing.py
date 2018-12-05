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


df = pd.read_csv('diabetes_dataset.csv')
df.head()
#print(df.shape)
#print(df.isnull().any().sum())

msno.matrix(df)
#%%
#df.isnull().sum()

imputer_glucose = Imputer(missing_values=np.nan, strategy='median', axis=0)
imputer_bloodpressure = Imputer(missing_values=np.nan, strategy='median', axis=0)
imputer_skinthickness = Imputer(missing_values=np.nan, strategy='median', axis=0)
imputer_insulin = Imputer(missing_values=np.nan, strategy='median', axis=0)
imputer_bmi = Imputer(missing_values=np.nan, strategy='median', axis=0)

df[['Glucose']] = imputer_glucose.fit_transform(df[['Glucose']])
df[['BloodPressure']] = imputer_bloodpressure.fit_transform(df[['BloodPressure']])
df[['SkinThickness']] = imputer_skinthickness.fit_transform(df[['SkinThickness']])
df[['Insulin']] = imputer_insulin.fit_transform(df[['Insulin']])
df[['BMI']] = imputer_bmi.fit_transform(df[['BMI']])



df.isnull().sum()

#%%
correlation_matrix = df.corr()
fig = plt.figure(figsize=(12,9))
sns.heatmap(correlation_matrix, vmax=0.8, square = True)
plt.show()

#%%
df['Pregnancies'] = StandardScaler().fit_transform(df['Glucose'].values.reshape(-1,1))
df['Glucose'] = StandardScaler().fit_transform(df['Glucose'].values.reshape(-1,1))
df['BloodPressure'] = StandardScaler().fit_transform(df['BloodPressure'].values.reshape(-1,1))
df['SkinThickness'] = StandardScaler().fit_transform(df['Glucose'].values.reshape(-1,1))
df['Insulin'] = StandardScaler().fit_transform(df['Glucose'].values.reshape(-1,1))
df['BMI'] = StandardScaler().fit_transform(df['Glucose'].values.reshape(-1,1))
df['DiabetesPedigreeFunction'] = StandardScaler().fit_transform(df['Glucose'].values.reshape(-1,1))
df['Age'] = StandardScaler().fit_transform(df['Glucose'].values.reshape(-1,1))

#df = df.drop(['Glucose', 'BloodPressure'], axis = 1)
df.head()
#imputer = Imputer(missing_values=np.nan, strategy='median', axis=0)
#df[['fnlwgt']] = imputer.fit_transform(df[['fnlwgt']])