# -*- coding: utf-8 -*-
"""
@Autor: Thiago Aparecido Gonçalves da Costa
@Disciplina: Redes Neurais

**********************Exercício 2**********************
Considere o problema de auto-associador (encoding problem) no qual um
conjunto de padrões ortogonais de entrada são mapeados num conjunto de
padrões de saída ortogonais através de uma camada oculta com um número
pequeno de neurônios.

Essencialmente, o problema é aprender uma codificação de padrão com p-bit em
um padrão de log2 p-bit, e em seguida aprender a decodificar esta representação
num padrão de saída.

Construir o mapeamento gerado pelo algoritmo backpropagation para o caso do
mapeamento identidade dado por:
    * Padrão de entrada: Id(10x10)
    * Padrão de saída: Id(10x10)
        Onde Id ​denota a matriz identidade

**********************TREINAMENTO**********************

Matriz identidade 10x10:
    
1 0 0 0 0 0 0 0 0 0
0 1 0 0 0 0 0 0 0 0
0 0 1 0 0 0 0 0 0 0 
0 0 0 1 0 0 0 0 0 0 
0 0 0 0 1 0 0 0 0 0
0 0 0 0 0 1 0 0 0 0
0 0 0 0 0 0 1 0 0 0
0 0 0 0 0 0 0 1 0 0
0 0 0 0 0 0 0 0 1 0
0 0 0 0 0 0 0 0 0 1 
"""
import numpy as np
import math as mth
from random import uniform
import pandas as pd

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
            vetor_oculta = np.append(vetor_oculta,uniform(-0.5,0.5))
            cont+=1
         
        aux = vetor_oculta.reshape(self.oculta, (self.entrada+1))
        return np.asmatrix(aux)
        
    #**********************INICIAR O MODELO DA CAMADA DE SAÍDA******
    def initOutputModel(self):
        
        #************PESOS E THETAS PARA A CAMADA DE SAÍDA**********
        vetor_saida = np.array([])
        cont = 0
        while(cont < (self.saida * (self.oculta+1))):
            vetor_saida = np.append(vetor_saida,uniform(-0.5,0.5))
            cont+=1
         
        aux = vetor_saida.reshape(self.saida, (self.oculta+1))
        return np.asmatrix(aux)

#**********************FNET = ONDE GUARDO OS RESULTADOS**********************
class Fnet():

    net_hidden = []
    fnet_hidden = []
    net_output = []
    fnet_output = []


##########################################################################################


# NET = SOMATÓRIA DOS PESOS PELA ENTRADA
def MLP_Forward(model, teste, fnet):
	
	#**********VETOR DE ENTRADAS**********
    teste = np.squeeze(np.asarray(teste))
    teste = np.append(teste,1)
	
	#**********VETOR DE ENTRADAS**********
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

def MLP_Backpropagation(model, dataset, eta = 0.5, limiar = 0.001, epocas = 100000):

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
				
				
                limiar_backpropagation = limiar_backpropagation + np.sum(np.power(error,2))
                
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
                
                model.output_model = model.output_model+eta*(np.transpose(delta_saida)*fnet1)

				
                x1=np.squeeze(np.asarray(entradas_desejadas))
                x1=np.append(x1,1)
                x1=np.asmatrix(x1)
                
     				 #Treinamento da camada escondida
                # w(t+1) = w(t) - eta * delta_h * Xp
                model.hidden_model = model.hidden_model+eta*(np.transpose(delta_oculta)*x1)

                limiar_backpropagation = limiar_backpropagation / dataset.A.shape[0]
                epocas += 1
            
            print('Erro Médio Quadrático = ',limiar_backpropagation)
            cont_epocas +=1
            
        print('--------------TERMINOU--------------')
        print('Quantidade de Épocas = ',cont_epocas)
        
        if(cont_epocas < epocas):
            print('--------------TREINADO--------------')
        
        return(model, resultados)

##########################################################################################
dataset_treinamento = pd.read_csv('dataset_treinamento.csv')
dataset_teste = pd.read_csv('dataset_teste.csv')

qtd_entradas = 10
qtd_ocultas =  int(np.around(mth.log2(qtd_entradas)))
qtd_saidas = 10

mlp = MLP_Arquitetura(qtd_entradas,qtd_ocultas,qtd_saidas)

dataset_treinamento=np.asmatrix(dataset_treinamento)

modelo, resultados = MLP_Backpropagation(mlp, dataset_treinamento, eta = 0.5, limiar = 0.001, epocas = 100000)

print('*****************TESTE 1************************')
print('******************ENTRADA***********************')
for linha in range(dataset_treinamento.A.shape[0]):
    print(dataset_treinamento[linha,0:mlp.entrada]) 

print('************************************************')

print('***************SAÍDA DESEJADA*******************')
for linha in range(dataset_treinamento.A.shape[0]):
    print(dataset_treinamento[linha,mlp.entrada:dataset_treinamento.A.shape[1]]) 

print('************************************************')

print('***************SAÍDA TREINADA*******************')
for linha in range(dataset_treinamento.A.shape[0]):
    aux = MLP_Forward(modelo,dataset_treinamento[linha,0:mlp.entrada],resultados)
    print(np.around(np.transpose(aux.fnet_output))) 

print('************************************************')

print('************TESTE 2 - NUNCA VISTO***************')

dataset_teste=np.asmatrix(dataset_teste)

print('******************ENTRADA***********************')
for linha in range(dataset_teste.A.shape[0]):
    print(dataset_teste[linha,0:mlp.entrada]) 

print('***************SAÍDA TREINADA*******************')
for linha in range(dataset_teste.A.shape[0]):
    aux = MLP_Forward(modelo,dataset_teste[linha,0:mlp.entrada],resultados)
    print(np.around(np.transpose(aux.fnet_output))) 

