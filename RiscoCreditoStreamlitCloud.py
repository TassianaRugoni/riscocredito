# -*- coding: utf-8 -*-
"""
Created on Wed May 22 20:32:16 2024

@author: Cliente
"""

"""
# Problema/Hipótese
Diminuição de Inadimplência
"""

#Bibliotecas
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#Base Full
basefull = pd.read_csv('RiscodeCredito.csv', sep=',')
dffull = pd.DataFrame(basefull)
dffull = dffull.dropna()
dffull

#Base Amostral
df = dffull.head(100)


#Histograma da Idade
st.write('Histograma')
fig, ax = plt.subplots(figsize=(10, 2))
sns.histplot(data=dffull, x="idade", element="step")
st.pyplot(fig) 

#Boxplot da Idade
st.write('Boxplot')
fig, ax = plt.subplots(figsize=(10, 2))
sns.boxplot(x=basefull["idade"])
st.pyplot(fig)

#Separação das Variáveis 
x = df[['idade']]
y = df[['meses_de_relacionamento']]

#Machine Learning - Regressão Logística
from sklearn import linear_model

#Função para o Cálculo da Regressão Linear - Equação da Reta
modelo =  linear_model.LinearRegression()

#Treino do Modelo
modelo.fit(x, y)

#Gráfico de Dispersão com a melhor reta
st.write('Gráfico de Dispersão - Regressão Linear')
fig, ax = plt.subplots(figsize=(10, 6))
sns.regplot(x='idade', y='meses_de_relacionamento', data=df, ax=ax, line_kws={'color': 'red'})
st.pyplot(fig)