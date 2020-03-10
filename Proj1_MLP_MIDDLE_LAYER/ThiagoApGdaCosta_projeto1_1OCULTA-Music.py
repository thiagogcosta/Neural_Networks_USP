# -*- coding: utf-8 -*-
"""
@Autor: Thiago Aparecido Gonçalves da Costa
@Disciplina: Redes Neurais

DATASET = MUSIC
QUANTIDADE DE CAMADAS OCULTAS = 1

"""
import numpy as np
import sklearn as sk
import sklearn.utils as sk_utils
from sklearn.model_selection import train_test_split
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

##########################################################################################

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
                
                limiar
                limiar_backpropagation += np.sum(np.power(error,2))
                
                limiar_backpropagation = limiar_backpropagation / dataset.A.shape[0]
                
                
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
                
                
            #print('Erro Médio Quadrático = '+str(limiar_backpropagation)+'     Epocas: '+str(cont_epocas))
            
            cont_epocas +=1
            
            
            
        print('--------------TERMINOU--------------')
        print('Erro Médio Quadrático = '+str(limiar_backpropagation)+'     Epocas: '+str(cont_epocas))
        #print('Quantidade de Épocas = ',cont_epocas)
        
        if(cont_epocas < epocas):
            print('--------------TREINADO--------------')
        
        return(model, resultados, cont_epocas, limiar_backpropagation)

##########################################################################################

dataset_output = pd.DataFrame(columns=['cam_ocultas','cam_1','parcela_teste','taxa_aprendizado', 'erro_quadratico', 'limite_epocas','cont_epocas'])

#*************EMBARALHANDO E SELECIONANDO PARCELAS DA BASE DE DADOS*************
v = [20,40,68]
t = [0.4,0.3]
eta = [0.4,0.6]
qtd_entradas = 68
qtd_saidas = 2
epocas = [2000, 10000]

    
for epoc in epocas:
     
    for i in v:
        for k in t:
            for j in eta:
                
                #********************************NORMALIZAÇÃO MIN-MAX***************************

                dataset = pd.read_csv('features.csv')
                
                normalizer = sk.preprocessing.MinMaxScaler(feature_range = (0,1))
                dataset_normalizado = normalizer.fit_transform(dataset)
                
                dataset_normalizado = pd.DataFrame(dataset_normalizado, columns= ['E1','E2','E3','E4','E5','E6','E7','E8','E9','E10','E11','E12','E13','E14','E15','E16','E17','E18','E19','E20','E21','E22','E23','E24','E25','E26','E27','E28','E29','E30','E31','E32','E33','E34','E35','E36','E37','E38','E39','E40','E41','E42','E43','E44','E45','E46','E47','E48','E49','E50','E51','E52','E53','E54','E55','E56','E57','E58','E59','E60','E61','E62','E63','E64','E65','E66','E67','E68','S1','S2'])


                d_embaralhado = sk_utils.shuffle(dataset_normalizado)
                
                treinamento, teste = train_test_split(d_embaralhado, test_size=k)
                   
                #******************************************************************************* 
                qtd_ocultas = i
                
                    
                #*******EMBARALHO O DATASET DE TREINAMENTO PARA NÃO DECORAR A ORDEM*********
                mlp = MLP_Arquitetura(qtd_entradas,qtd_ocultas,qtd_saidas)
                
                dataset_treinamento=np.asmatrix(treinamento)
                dataset_teste=np.asmatrix(teste)
               
                limiar = 0.0001
                
                modelo, resultados, cont_epocas, limiar_backpropagation = MLP_Backpropagation(mlp, dataset_treinamento, j, limiar, epoc)
                
                print('Qtd de camadas ocultas: 1')
                print('Qtd de neurônios na camada oculta: '+str(qtd_ocultas))
                print('Tamanho de to teste: '+str(k))
                print('Limite epocas: '+str(epoc))
                print('Eta: '+str(j))
                print('Erro quadrático médio: '+str(limiar_backpropagation))
                print('cont_epocas: '+str(cont_epocas))
                
                df_aux = pd.DataFrame([['1', qtd_ocultas, j, k, limiar_backpropagation, epoc, cont_epocas]], columns=['cam_ocultas','cam_1','parcela_teste','taxa_aprendizado', 'erro_quadratico', 'limite_epocas','cont_epocas'])
                dataset_output = pd.concat([dataset_output, df_aux])
                    
                    
dataset_output.to_csv('1OCULTA_MUSIC.csv')
