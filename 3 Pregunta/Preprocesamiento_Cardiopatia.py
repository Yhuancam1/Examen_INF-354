# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 17:06:49 2023

@author: HP
"""
import numpy as np
import pandas as pd

datos = pd.read_csv('C:/Users/HP/Desktop/Dataset_Cardiopatia_00.csv')

x = datos.iloc[:,5:7].values
y = datos.iloc[:,3].values

from sklearn.impute import SimpleImputer

# Crear una instancia de SimpleImputer para rellenar con la mediana
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Rellenar los valores faltantes en el conjunto de datos 'X'
imputer = imputer.fit(x)
X_imputed = imputer.transform(x)

print("Conjunto de datos con valores faltantes rellenados con la media:")
print(X_imputed)



from sklearn.preprocessing import LabelEncoder

# Datos de ejemplo
etiquetas = datos.iloc[:,10].values

# Crear una instancia de LabelEncoder
codificador = LabelEncoder()

# Ajustar y transformar los datos
etiquetas_codificadas = codificador.fit_transform(etiquetas)

# Imprimir las etiquetas codificadas
print("Etiquetas originales: ", etiquetas)
print("Etiquetas codificadas: ", etiquetas_codificadas)


labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)