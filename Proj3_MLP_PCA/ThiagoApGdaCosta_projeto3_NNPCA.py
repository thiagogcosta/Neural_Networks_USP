'''
@Autor: Thiago Aparecido Gonçalves da Costa
@Disciplina: Redes Neurais

**********************PROJETO 3**********************

DATASET = WINE
TIPO DE REDE: PCA ADAPTATIVA
'''
import numpy as np
import sklearn as sk
import sklearn.utils as sk_utils
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from random import uniform
import pandas as pd
import decimal


#**************************PCA NETWORK*******************************
def NN_PCA(X_wine, Y_wine, reducao_dimensionalidade, taxa_aprendizado, epocas, ciclos):

     #*********************NORMALIZAÇÃO*********************
    X_wine = StandardScaler().fit_transform(X_wine)

    # Weight matrices
    W = np.random.uniform(-0.01, 0.01, size=(X_wine.shape[1], reducao_dimensionalidade))
    V = np.triu(np.random.uniform(-0.01, 0.01, size=(reducao_dimensionalidade, reducao_dimensionalidade)))
    np.fill_diagonal(V, 0.0)

    # Training process
    for e in range(epocas):
        for i in range(X_wine.shape[0]):
            y_p = np.zeros((reducao_dimensionalidade, 1))
            xi = np.expand_dims(X_wine[i], 1)

            for _ in range(ciclos):
                y = np.dot(W.T, xi) + np.dot(V, y_p)
                y_p = y.copy()

            dW = np.zeros((X_wine.shape[1], reducao_dimensionalidade))
            dV = np.zeros((reducao_dimensionalidade, reducao_dimensionalidade))

            for t in range(reducao_dimensionalidade):
                y2 = np.power(y[t], 2)
                dW[:, t] = np.squeeze((y[t] * xi) + (y2 * np.expand_dims(W[:, t], 1)))
                dV[t, :] = -np.squeeze((y[t] * y) + (y2 * np.expand_dims(V[t, :], 1)))

            W += (taxa_aprendizado * dW)
            V += (taxa_aprendizado * dV)

            V = np.tril(V)
            np.fill_diagonal(V, 0.0)

            W /= np.linalg.norm(W, axis=0).reshape((1, reducao_dimensionalidade))


    # Compute all output components
    Y_comp = np.zeros((X_wine.shape[0], reducao_dimensionalidade))


    for i in range(X_wine.shape[0]):
        y_p = np.zeros((reducao_dimensionalidade, 1))
        xi = np.expand_dims(X_wine[i], 1)

        for _ in range(ciclos):

            Y_comp[i] = np.squeeze(np.dot(V.T, y_p) + np.dot(W.T, xi))
            y_p = np.asmatrix(Y_comp[i]).T.copy()

    return Y_comp

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

########################---------MLP---------########################

#**********************FUNÇÃO DE ATIVAÇÃO***************************
def getFnet(net):
    return (1 / (1 + np.exp(-net)))

#**********************DERIVADA DA FUNÇÃO DE ATIVAÇÃO***************
def getDerFnet(fnet):
    return (np.multiply(fnet, (1 - fnet)))

#**********************MODELO INICIAL DA ARQUITETURA****************
class MLP_Arquitetura:

    def __init__(self, entrada, oculta, saida):

        self.entrada = entrada
        self.oculta = oculta
        self.saida = saida
        self.hidden_model = self.initHiddenModel()
        self.output_model = self.initOutputModel()

    #**********************INICIAR O MODELO DA CAMADA OCULTA********
    def initHiddenModel(self):

        #************PESOS E THETAS PARA A CAMADA OCULTA************
        vetor_oculta = np.array([])
        cont = 0
        while(cont < (self.oculta * (self.entrada+1))):
            vetor_oculta = np.append(vetor_oculta,uniform(-1,1))
            cont+=1

        aux = vetor_oculta.reshape(self.oculta, (self.entrada+1))
        return np.asmatrix(aux)

    #**********************INICIAR O MODELO DA CAMADA DE SAÍDA******
    def initOutputModel(self):

        #************PESOS E THETAS PARA A CAMADA DE SAÍDA**********
        vetor_saida = np.array([])
        cont = 0
        while(cont < (self.saida * (self.oculta+1))):
            vetor_saida = np.append(vetor_saida,uniform(-1,1))
            cont+=1

        aux = vetor_saida.reshape(self.saida, (self.oculta+1))
        return np.asmatrix(aux)

#**********************FNET = ONDE GUARDO OS RESULTADOS**********************
class Fnet():

    net_hidden = []
    fnet_hidden = []
    net_output = []
    fnet_output = []

# NET = SOMATÓRIA DOS PESOS PELA ENTRADA
def MLP_Forward(model, teste, fnet):

	#**********VETOR DE ENTRADAS**********
    teste = np.squeeze(np.asarray(teste))
    teste = np.append(teste,1)

	#*****VETOR DE ENTRADAS TRANSPOSTO*****
    hidden  = np.asmatrix(model.hidden_model)

    teste = np.asmatrix(teste)
    teste = np.transpose(teste)

    #********** SAÍDA DA CAMADA OCULTA**********
    net_hidden = hidden*teste

    fnet.net_hidden = net_hidden

    fnet_hidden = getFnet(net_hidden)
    fnet.fnet_hidden = fnet_hidden

    #*******************************************
    output  = model.output_model
    output = np.asmatrix(output)

    fnet_hidden = np.squeeze(np.asarray(fnet_hidden))
    fnet_hidden = np.append(fnet_hidden,1)
    fnet_hidden = np.asmatrix(fnet_hidden)
    fnet_hidden = np.transpose(fnet_hidden)

    #********** SAÍDA DA CAMADA DE SAÍDA**********
    net_output = output*fnet_hidden

    fnet.net_output = net_output


    fnet_output = getFnet(net_output)

    fnet.fnet_output = fnet_output
    #*******************************************

    return fnet

def MLP_Backpropagation(model, dataset, eta, limiar, epocas):

        limiar_backpropagation = 2 * limiar
        cont_epocas = 0
        resultados = Fnet()

        while(limiar_backpropagation > limiar and cont_epocas <= epocas):

            limiar_backpropagation = 0
            dataset=np.asmatrix(dataset)

            for linha in range(dataset.A.shape[0]):

                entradas_desejadas = dataset[linha,0:model.entrada]
                saidas_desejadas =   dataset[linha,model.entrada:dataset.A.shape[1]]

                #*********************FORWARD**********************
                resultados = MLP_Forward(model , entradas_desejadas,resultados)

                #*********************ERROS************************
                error = saidas_desejadas - np.transpose(resultados.fnet_output)

                #limiar_local = np.sum(np.power(error,2))

                limiar_backpropagation += np.sum(np.power(error,2))

                #print(limiar_backpropagation)

                #*********************DELTA DA SAÍDA************************
				    # delta_o = (Yp - Op) * f_o_p'(net_o_p)
                delta_saida = np.multiply(error , np.transpose(getDerFnet(resultados.fnet_output)))

                #***********************************************************

                #*********************DELTA DA OCULTA***********************
                # delta_hidden = f_h_p'(net_h_p) * sum =(delta_o * w_o_kj)
                w_o=np.asmatrix(model.output_model[:,0:model.oculta])

                delta_oculta = np.multiply(np.transpose(getDerFnet(resultados.fnet_hidden)),(delta_saida*w_o))

				    #*********************TREINAMENTO***************************
                fnet1 = np.squeeze(np.asarray(resultados.fnet_hidden))
                fnet1 = np.append(fnet1,1)
                fnet1 = np.asmatrix(fnet1)

                # Treinamento da camada de saída
                # w(t+1) = w(t) + eta * dE2_o * i_pj

                #************************momentum * modelo***************************

                #print(model.output_model)

                '''aux = model.output_model.copy()

                aux.fill(momentum)

                #print(aux)

                aux_model = np.multiply(model.output_model,aux)'''

                #print(aux_model)

                aux_eta = eta*(np.transpose(delta_saida)*fnet1)

                #print(aux_eta)

                model.output_model = model.output_model+aux_eta

                #print(model.output_model)



                x1=np.squeeze(np.asarray(entradas_desejadas))
                x1=np.append(x1,1)
                x1=np.asmatrix(x1)

     		    #Treinamento da camada escondida
                # w(t+1) = w(t) - eta * delta_h * Xp

                #************************momentum * modelo***************************

                '''aux = model.hidden_model.copy()

                aux.fill(momentum)

                #print(aux)

                aux_model = np.multiply(model.hidden_model,aux)'''


                aux_eta = eta*(np.transpose(delta_oculta)*x1)

                model.hidden_model = model.hidden_model+aux_eta


                limiar_backpropagation = limiar_backpropagation / dataset.A.shape[0]


            #print('Erro Médio Quadrático = ',limiar_backpropagation)
            cont_epocas +=1



        print('--------------TERMINOU--------------')
        print('Quantidade de Épocas = ',cont_epocas)

        if(cont_epocas < epocas):
            print('--------------TREINADO--------------')

        return(model, resultados, cont_epocas)

##########################################################################################


#*****************WINE*****************
#dataset = pd.read_csv('wine.csv')
wine = datasets.load_wine()

X_wine = wine.data
Y_wine = wine.target
target_names_wine  = wine.target_names


reducao_dimensionalidade = 2
taxa_aprendizado = 0.001
epocas = 100
ciclos = 5

#*****************PCA WINE*****************
resultado_pcaClassico = PCA(X_wine, reducao_dimensionalidade)

#*****************VISUALIZAÇÃO*****************
Visualizacao('WINE: NOT PCA CLASSICAL',X_wine, Y_wine, target_names_wine)
Visualizacao('WINE: PCA CLASSICAL',resultado_pcaClassico, Y_wine, target_names_wine)

#*****************REDE PCA WINE*****************
resultado_pcaAdaptativo = NN_PCA(X_wine, Y_wine, reducao_dimensionalidade, taxa_aprendizado, epocas, ciclos)

#*****************VISUALIZAÇÃO*****************
Visualizacao('WINE: NOT PCA NETWORK',X_wine, Y_wine, target_names_wine)
Visualizacao('WINE: PCA NETWORK',resultado_pcaAdaptativo, Y_wine, target_names_wine)

##########################################################################################

dataset_pcaClassico = pd.DataFrame(columns= ['E1','E2','S1'])

for x in range(len(Y_wine)):
    dataset_pcaClassico.loc[x, 'E1'] = resultado_pcaClassico[x][0]
    dataset_pcaClassico.loc[x, 'E2'] = resultado_pcaClassico[x][1]
    dataset_pcaClassico.loc[x, 'S1'] = Y_wine[x]

dataset_pcaAdaptativo = pd.DataFrame(columns= ['E1','E2','S1'])

for x in range(len(Y_wine)):
    dataset_pcaAdaptativo.loc[x, 'E1'] = resultado_pcaAdaptativo[x][0]
    dataset_pcaAdaptativo.loc[x, 'E2'] = resultado_pcaAdaptativo[x][1]
    dataset_pcaAdaptativo.loc[x, 'S1'] = Y_wine[x]

#QUANTIDADE DE NEURÔNIOS NAS CAMADAS
qtd_entradas = 2
qtd_saidas = 3
qtd_ocultas = 2

#TAXA DE APRENDIZADO
eta = 0.5

#LIMITE MÁXIMO DE ÉPOCAS
epocas = 10000

#LIMITE MÁXIMO DE ÉPOCAS
porc_teste = 0.3

#LIMIAR
limiar = 0.00001

vector_dataset = [dataset_pcaClassico, dataset_pcaAdaptativo]

contador_dataset = 0
for dataset in vector_dataset:
    #*********************************************PRÉ-PROCESSAMENTO************************************

    dataset_bin = pd.DataFrame(columns= ['E1','E2','S1','S2','S3'])

    labels = set(Y_wine)

    for x in labels:

        d = dataset[dataset['S1'] == x]

        if(x == 1):
            d.loc[:,'S1'] = 0
            d.loc[:,'S2'] = 0
            d.loc[:,'S3'] = 1

        elif (x == 2):
            d.loc[:,'S1'] = 0
            d.loc[:,'S2'] = 1
            d.loc[:,'S3'] = 0

        else:
            d.loc[:,'S1'] = 1
            d.loc[:,'S2'] = 0
            d.loc[:,'S3'] = 0

        dataset_bin = pd.concat([dataset_bin, d])

    #********************************NORMALIZAÇÃO ENTRE 0 E 1 DA BASE DE DADOS***************************

    normalizer = sk.preprocessing.MinMaxScaler(feature_range = (0,1))
    dataset_normalizado = normalizer.fit_transform(dataset_bin)
    dataset_normalizado = pd.DataFrame(dataset_normalizado, columns= ['E1','E2','S1','S2','S3'])

    #**************************************************EMBARALHANDO**************************************

    dataset_teste = pd.DataFrame(columns= ['E1','E2','S1','S2','S3'])
    dataset_treinamento = pd.DataFrame(columns= ['E1','E2','S1','S2','S3'])

    dataset_normalizado = sk_utils.shuffle(dataset_normalizado)

    #********************************SELECIONANDO A PARCELA PARA TREINAMENTO E TESTE**********************

    saidas = ['S1','S2','S3']

    for x in saidas:

        d = dataset_normalizado[dataset_normalizado[x] == 1]

        treinamento, teste = train_test_split(d, test_size = porc_teste)

        dataset_teste = dataset_teste.append(teste)

        dataset_treinamento = dataset_treinamento.append(treinamento)


    mlp = MLP_Arquitetura(qtd_entradas,qtd_ocultas,qtd_saidas)

    dataset_treinamento=np.asmatrix(dataset_treinamento)
    dataset_teste=np.asmatrix(dataset_teste)

    modelo, resultados, cont_epocas = MLP_Backpropagation(mlp, dataset_treinamento, eta, limiar , epocas)

    #*******************************************ACURÁCIA**********************************************
    somatoria = 0
    for linha in range(dataset_teste.A.shape[0]):
        aux_sd = dataset_teste[linha,mlp.entrada:dataset_teste.A.shape[1]]
        aux_sd = np.squeeze(np.asarray(aux_sd))

        aux_saida = MLP_Forward(modelo,dataset_teste[linha,0:mlp.entrada],resultados)
        aux_saida = np.squeeze(np.asarray(aux_saida.fnet_output))
        aux_saida =  np.around(aux_saida)

        #COMPARAÇÃO SE O VETOR DE SAÍDA É IGUAL AO DESEJADO
        sum_equal = 0
        for i in range(len(aux_sd) - 1):
            if(aux_sd[i] != aux_saida[i]):
                sum_equal = 1
                break
        #CASO O VETOR DE SAÍDA SEJA IGUAL, LOGO SOMO 1
        if(sum_equal == 0):
            somatoria +=1

        #print('Saida Desejada: '+str(aux_sd)+' Saida Obtida: '+str(aux_saida))

    # ACURACIA É A RAZÃO ENTRE A "SOMATORIA" E A QUANTIDADE DE EXEMPLOS DE TESTES
    acuracia = (somatoria / dataset_teste.A.shape[0]) * 100

    d = decimal.Decimal(acuracia)
    acuracia = round(d,2)

    #***************************************************************************************************

    if(contador_dataset == 0):
        print('MLP com PCA Classico')
    else:
        print('MLP com PCA Adaptativo')
    print('Qtd de neurônios na camada de entrada: '+str(qtd_entradas))
    print('Qtd de neurônios na camada oculta: '+str(qtd_ocultas))
    print('Qtd de neurônios na camada de saída: '+str(qtd_saidas))
    print('Parcela de teste: '+str(porc_teste))
    print('Taxa de aprendizado: '+str(eta))
    print('Acurácia: '+str(acuracia))
    print('Quantidade de epocas: '+str(cont_epocas))

    contador_dataset += 1
