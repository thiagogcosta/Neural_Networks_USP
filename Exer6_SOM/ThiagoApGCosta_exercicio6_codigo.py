
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

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
def Visualizacao(nome_grafico,dados, classes, nomes_classes, som):

    colors = ['navy', 'turquoise', 'darkorange']
    lw = 2
    plt.figure()
    plt.title(nome_grafico)
    if(som == 1):
        for color, i, target_name in zip(colors, [1, 2, 3], nomes_classes):
            plt.scatter(dados[classes == i, 0], dados[classes == i, 1], color=color, alpha=.8, lw=lw, label=target_name)
    else:
        for color, i, target_name in zip(colors, [0, 1, 2], nomes_classes):
            plt.scatter(dados[classes == i, 0], dados[classes == i, 1], color=color, alpha=.8, lw=lw, label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.show()

# Sed para aleatorização
np.random.seed(2018)

#*****************WINE*****************

wine = pd.read_csv("wine.csv")

wine=(wine-wine.mean())/wine.std()

iteracoes = 3*len(wine.index)

#Numero de nós de saída
num_nodes = 3

tam_entrada = len(wine.columns)

taxa_aprendizado = 0.01

#Matriz de pesos
matriz_pesos = 4*np.random.rand(tam_entrada,num_nodes)-2

print ("Matriz de pesos:", matriz_pesos)

for iter in range(iteracoes):

    dist_bmu = float("inf")

    row_index = np.random.randint(len(wine.index))

    dado_escolhido = wine.loc[[row_index]]

    for node in range(num_nodes):

        dist_neuron = np.linalg.norm(dado_escolhido-matriz_pesos[:,node])

        if dist_neuron < dist_bmu:

            dist_bmu = dist_neuron

            weight_bmu = matriz_pesos[:,node]
            index_bmu = node

    learn_rate = taxa_aprendizado*np.exp(-iter/iteracoes)

    #Atualização de pesos (w_{t+1} = w_{t} + L(t)*(x_{i} - w_{t}))
    matriz_pesos[:,index_bmu] = np.add(weight_bmu,learn_rate*(np.subtract(dado_escolhido,weight_bmu)))


print ("Pesos treinados da SOM", matriz_pesos)

agrupamentos = np.zeros(len(wine.index))

for index, data in wine.iterrows():

    dist_cluster = float("inf")

    for centroid in range(num_nodes):

        dist_centroid = np.linalg.norm(data-matriz_pesos[:,centroid])

        if dist_centroid < dist_cluster:

                dist_cluster = dist_centroid

                agrupamentos[index] = centroid+1

# Adicionando os rótulos na coluna do dataset
wine["agrupamentos"] = agrupamentos

#********************EXECUÇÃO********************
vinho = datasets.load_wine()

X_vinho = vinho.data
Y_vinho = vinho.target
target_names_vinho = vinho.target_names

reducao_dimensionalidade = 2

#*****************PCA WINE*****************
resultado = PCA(X_vinho, reducao_dimensionalidade)

print(resultado)

#*****************VISUALIZAÇÃO*****************
Visualizacao('WINE NOT SOM: PCA',resultado, Y_vinho, target_names_vinho, 0)
Visualizacao('WINE SOM: PCA',resultado, wine['agrupamentos'].values, target_names_vinho, 1)

#*****************COMPARAÇÃO*****************
result_rotulos = wine['agrupamentos'].values
dif_rotulos = 0

if(len(Y_vinho) == len(result_rotulos)):
    for i in range(len(Y_vinho)):
        aux = 0
        if(Y_vinho[i] == 0):
            aux = 1
        elif(Y_vinho[i] == 1):
            aux = 2
        else:
            aux = 3
        if(aux != result_rotulos[i]):
            dif_rotulos +=1

print(dif_rotulos)
