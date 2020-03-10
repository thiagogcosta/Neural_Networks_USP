# -*- coding: utf-8 -*-
"""
Nome: Thiago Aparecido Gonçalves da Costa
Disciplina: Redes Neurais

**************************ATIVIDADE 5**************************

Atividade - 5 - PCA:

1 - Implementar e testar a técnica PCA clássica com o seguinte conjunto: wine.dat;
2 - Apresentar os resultados de forma que os agrupamentos possam ser visualizados;
3 - Utilize linguagem de programação Python;
4 - Elabore um relatório, de 1 página, descrevendo o trabalho e os resultados.

"""

import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#*****************PCA*****************
def PCA(dados, reducao_dimensionalidade):
    
    X = dados
    #*********************NORMALIZAÇÃO*********************
    wine_norm = StandardScaler().fit_transform(X)
    
    #***************MATRIZ DE COVARIÂNCIA*******************
    matriz_convariance = np.cov(wine_norm.T)
    
    
    #***********AUTOVETORES E AUTOVALORES******************
    auto_valores, auto_vetores = np.linalg.eig (matriz_convariance)
    
    #definindo um vetor com a ordem dos auto-valores de forma decrescente 
    orden_autoval = np.argsort(auto_valores)[::-1]
    
    #ordenando os auto-vetores conforme a ordem dos auto valores
    auto_vetores = auto_vetores[:,orden_autoval]
    
    #ordenando os auto-vetores conforme a ordem dos auto valores
    auto_valores = auto_valores[orden_autoval]
    
    #reduzindo a dimensionalidade
    auto_vetores = auto_vetores[:, :reducao_dimensionalidade]
    
    #transposta da multiplicação da transposta dos auto_vetores com a transposta dos dados normalizados
    resultado = np.dot(auto_vetores.T, wine_norm.T).T
    
    return resultado

#*****************VISUALIZAÇÃO*****************
def Visualizacao(nome_grafico,dados, classes, nomes_classes):

    colors = ['navy', 'turquoise', 'darkorange']
    lw = 2
    
    plt.figure()
    plt.title(nome_grafico)
    for color, i, target_name in zip(colors, [0, 1, 2], nomes_classes):
        plt.scatter(dados[classes == i, 0], dados[classes == i, 1], color=color, alpha=.8, lw=lw, label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.show()

#*****************DIRETÓRIO*****************
path = 'C://Users//usuario//Desktop//Redes Neurais//ThiagoApGdaCosta_exercicio5//'

#*****************WINE*****************
wine = datasets.load_wine()
#wine = pd.read_csv(path+'wine.csv')

X_wine = wine.data
Y_wine = wine.target
target_names_wine  = wine.target_names

#*****************IRIS*****************
iris = datasets.load_iris()

X_iris = iris.data
Y_iris = iris.target
target_names_iris = iris.target_names

reducao_dimensionalidade = 2

#*****************PCA WINE*****************
resultado = PCA(X_wine, reducao_dimensionalidade)

#*****************VISUALIZAÇÃO*****************
Visualizacao('WINE: NOT PCA',X_wine, Y_wine, target_names_wine)
Visualizacao('WINE: PCA',resultado, Y_wine, target_names_wine)

#*****************PCA IRIS*****************
resultado = PCA(X_iris, reducao_dimensionalidade)

#*****************VISUALIZAÇÃO*****************
Visualizacao('IRIS: NOT PCA',X_iris, Y_iris, target_names_iris)
Visualizacao('IRIS: PCA',resultado, Y_iris, target_names_iris)
