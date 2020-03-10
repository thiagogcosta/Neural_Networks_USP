# -*- coding: utf-8 -*-
"""
@Autor: Thiago Aparecido Gonçalves da Costa
@Disciplina: Redes Neurais

**********************EXERCÍCIO 3**********************

DATASET = SEEDS
TIPO DE REDE NEURAL: RBF
"""
import numpy as np
import sklearn.utils as sk_utils
from sklearn.model_selection import train_test_split
from random import uniform
from sklearn.preprocessing import minmax_scale
import pandas as pd
import decimal
import math
from sklearn.cluster import KMeans 

#--------------------------------------------------------
#---------------------FUNÇÃO RADIAL----------------------
#--------------------------------------------------------

#*****************Largura da Função Radial***************
#****Largura = (1/quantidade de cluster) * (Somatório da distância absoluta de um centro para o mais próximo) 
def largura_radial(clusters):
    
    media = 0
    
    for i in range(7):
        
        somatorio = 0
        
        dist01 = abs(np.linalg.norm(clusters[0][i]-clusters[1][i]))
        
        dist02 = abs(np.linalg.norm(clusters[0][i]-clusters[2][i]))
        
        if(dist01 >= dist02):
            somatorio += dist01
        else:
            somatorio += dist02
        
        dist10 = abs(np.linalg.norm(clusters[1][i]-clusters[0][i]))
        
        dist12 = abs(np.linalg.norm(clusters[1][i]-clusters[2][i]))
        
        if(dist10 >= dist12):
            somatorio += dist10
        else:
            somatorio += dist12
        
        dist20 = abs(np.linalg.norm(clusters[2][i]-clusters[0][i]))
        
        dist21 = abs(np.linalg.norm(clusters[2][i]-clusters[1][i]))
        
        if(dist20 >= dist21):
            somatorio += dist20
        else:
            somatorio += dist21
        
        media += (somatorio / 3)
        
    return (media/7)
        

#**********************GAUSSIANA*************************
def funcao_gaussiana(clusters,entradas, sigma):
   soma = 0
   for i in range(len(entradas)):
       soma += np.power((clusters[i] - entradas[i]),2)
   return np.exp(-soma / np.power(2 * sigma, 2))

#****************MULTIQUADRÁTICA INVERSA*****************  
def funcao_multiquadratica(clusters,entradas, sigma):
    soma = 0
    for i in range(len(entradas)):
       soma += np.power((clusters[i] - entradas[i]),2)
   
    return math.sqrt(soma + np.power(sigma,2))
#--------------------------------------------------------
#---------------------FUNÇÃO LINEAR----------------------
#--------------------------------------------------------

#**********************FUNÇÃO DE ATIVAÇÃO***************************
def getFnet(net):
    return (1 / (1 + np.exp(-net)))
    
#**********************DERIVADA DA FUNÇÃO DE ATIVAÇÃO***************
def getDerFnet(fnet):
    return (np.multiply(fnet, (1 - fnet)))

#--------------------------------------------------------
#-------------------------MODELO-------------------------
#--------------------------------------------------------

#**********************MODELO INICIAL DA ARQUITETURA****************
class RBF_Arquitetura:
    
    def __init__(self, oculta, saida):
        
        self.oculta = oculta
        self.saida = saida
        self.output_model = self.initOutputModel()
        
        
    #**********************INICIAR O MODELO DA CAMADA DE SAÍDA******
    def initOutputModel(self):
        
        #************PESOS E THETAS PARA A CAMADA DE SAÍDA**********
        vetor_saida = np.array([])
        cont = 0
        while(cont < (self.saida * (self.oculta+1))):
            vetor_saida = np.append(vetor_saida,uniform(0,1))
            cont+=1
         
        aux = vetor_saida.reshape(self.saida, (self.oculta+1))
        return np.asmatrix(aux)

#**********************FNET = ONDE GUARDO OS RESULTADOS**********************
class Fnet():

    net_output = []
    fnet_output = []

##########################################################################################

# NET = SOMATÓRIA DOS PESOS PELA ENTRADA
def RBF_Forward(model, teste, clusters, sigma, fnet):
    
    vetor_gauss = []
    val_gauss = 0
    
    for i in range(len(clusters)):
        
        val_gauss = funcao_multiquadratica(np.squeeze(np.asarray(clusters[i])), teste, sigma)
        
        vetor_gauss = np.append(vetor_gauss, val_gauss)

	
    #**********VETOR DE ENTRADAS**********
    teste = np.append(vetor_gauss,1)
    
    vetor_gauss_teste = teste
    
    teste = np.asmatrix(teste)
    teste = np.transpose(teste)
    
    output  = model.output_model
    output = np.asmatrix(output)

    #********** SAÍDA DA CAMADA DE SAÍDA**********
    net_output = output*teste
    
    fnet.net_output = net_output      
    
    fnet_output = getFnet(net_output)
    
    
    fnet.fnet_output = fnet_output
    #*******************************************

    return fnet, vetor_gauss_teste

def RBF_Backpropagation(model, dataset, clusters, sigma, eta, limiar, epocas):

        limiar_backpropagation = 2 * limiar
        cont_epocas = 0
        resultados = Fnet()
		
        while(limiar_backpropagation > limiar and cont_epocas <= epocas):
            
            limiar_backpropagation = 0
            dataset=np.asmatrix(dataset)

            for linha in range(dataset.A.shape[0]):
                
                entradas_desejadas = np.squeeze(np.asarray(dataset[linha,0:7])) 
                                
                saidas_desejadas =   dataset[linha,7:dataset.A.shape[1]]
                 
                #*********************FORWARD**********************
                resultados, vetor_gauss_teste = RBF_Forward(model,entradas_desejadas, clusters, sigma, resultados)
                
                #*********************ERROS************************
                error = saidas_desejadas - np.transpose(resultados.fnet_output)
                
                limiar_backpropagation += np.sum(np.power(error,2))
                
                #*********************DELTA DA SAÍDA************************
				    # delta_o = (Yp - Op) * f_o_p'(net_o_p)
                delta_saida = np.multiply(error , np.transpose(getDerFnet(resultados.fnet_output)))
                
                #***********************************************************
                
                aux_eta = eta*(np.transpose(delta_saida) * vetor_gauss_teste)
                
                #*****INCLUSÃO DO TERMO MOMENTUM*****
                
                #*****MODELO OUTPU ANTIGO*****
                modelo_output_antigo  = model.output_model
                
                #*****MODELO OUTPUT NOVO*****
                modelo_output_novo  = model.output_model+aux_eta
                
                #*****INCLUSÃO DO TERMO MOMENTUM E ATUALIZAÇÃO DOS PESOS*****
                model.output_model = model.output_model + (modelo_output_novo - modelo_output_antigo) * 0.5
            		
                limiar_backpropagation = limiar_backpropagation / dataset.A.shape[0]

            print('Erro Médio Quadrático = ',limiar_backpropagation)
            cont_epocas +=1
        
        print('--------------TERMINOU--------------')
        print('Quantidade de Épocas = ',cont_epocas)
        
        if(cont_epocas < epocas):
            print('--------------TREINADO--------------')
        
        return(model, resultados, cont_epocas)

###################################################################################################
    
#****************************************LEITURA DA BASE DE DADOS**********************************
dataset = pd.read_csv('seeds.csv')

labels = dataset['S1'].values
labels = set(labels)

#QUANTIDADE DE NEURÔNIOS NAS CAMADAS
qtd_saidas = 3
qtd_ocultas = 3

#TAXA DE APRENDIZADO
eta = 0.5

#LIMITE MÁXIMO DE ÉPOCAS
epocas = 20000

#LIMITE MÁXIMO DE ÉPOCAS
porc_teste = 0.3

#LIMIAR 
limiar = 0.00001

#*********************************************PRÉ-PROCESSAMENTO*******************************************************
dataset_bin = pd.DataFrame(columns= ['E1','E2','E3','E4','E5','E6','E7','S1','S2','S3'])

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

dataset_normalizado = minmax_scale(dataset_bin, feature_range=(0,1), axis=0, copy=True )

d_norm = dataset_normalizado

dataset_normalizado = pd.DataFrame(dataset_normalizado, columns= ['E1','E2','E3','E4','E5','E6','E7','S1','S2','S3'])

#**************************************************EMBARALHANDO**************************************
dataset_teste = pd.DataFrame(columns= ['E1','E2','E3','E4','E5','E6','E7','S1','S2','S3'])
dataset_treinamento = pd.DataFrame(columns= ['E1','E2','E3','E4','E5','E6','E7','S1','S2','S3'])

dataset_normalizado = sk_utils.shuffle(dataset_normalizado)

#********************************SELECIONANDO A PARCELA PARA TREINAMENTO E TESTE**********************
saidas = ['S1','S2','S3']

dataset_normalizado.to_csv('dataset.csv')

for x in saidas:

    d = dataset_normalizado[dataset_normalizado[x] == dataset_normalizado['S1'].max()]    
    
    treinamento, teste = train_test_split(d, test_size=porc_teste)
    
    dataset_teste = dataset_teste.append(teste)
    
    dataset_treinamento = dataset_treinamento.append(treinamento)


mlp = RBF_Arquitetura(qtd_ocultas,qtd_saidas)

dataset_treinamento=np.asmatrix(dataset_treinamento)
dataset_teste=np.asmatrix(dataset_teste)

#**********************************APLICANDO O K-MEANS*************************************
#*************************************OBSERVAÇÃO:****************************************************
#***O estagiário PAE 
#****************************************************************************************************
#****************************************************************************************************
dataset_kmeans = d_norm[:,0:7]

kmedia = KMeans(n_clusters=3)  
resultado = kmedia.fit(dataset_kmeans) 

clusters = resultado.cluster_centers_

sigma = largura_radial(clusters)

modelo, resultados, cont_epocas = RBF_Backpropagation(mlp, dataset_treinamento, clusters, sigma, eta, limiar , epocas)

#*******************************************ACURÁCIA**********************************************
somatoria = 0

for linha in range(dataset_teste.A.shape[0]):
    aux_sd = dataset_teste[linha,7:dataset_teste.A.shape[1]]
    aux_sd = np.squeeze(np.asarray(aux_sd))

    #############################################################################
    
    dataset_teste=np.asmatrix(dataset_teste)
    
    entradas_desejadas = np.squeeze(np.asarray(dataset_teste[linha,0:7]))
         
    saidas_desejadas = dataset_teste[linha,7:dataset_teste.A.shape[1]]
    
    #############################################################################
    aux_saida, aux = RBF_Forward(modelo,entradas_desejadas,clusters, sigma, resultados)
    aux_saida = np.squeeze(np.asarray(aux_saida.fnet_output))       
    aux_saida =  np.around(aux_saida)
    
    print('Valor desejado: '+str(aux_sd)+ 'Valor obtido: '+str(aux_saida))
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
print('Qtd de neurônios na camada oculta: '+str(qtd_ocultas))
print('Qtd de neurônios na camada de saída: '+str(qtd_saidas))
print('Parcela de teste: '+str(porc_teste))
print('Taxa de aprendizado: '+str(eta))
print('Acurácia: '+str(acuracia))
print('Quantidade de epocas: '+str(cont_epocas))
