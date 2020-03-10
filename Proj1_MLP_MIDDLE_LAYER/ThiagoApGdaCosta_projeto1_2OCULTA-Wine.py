# -*- coding: utf-8 -*-
"""
@Autor: Thiago Aparecido Gonçalves da Costa
@Disciplina: Redes Neurais

**********************PROJETO 1**********************

DATASET = WINE
QUANTIDADE DE CAMADAS OCULTAS = 2

"""
import numpy as np
import sklearn as sk
import sklearn.utils as sk_utils
from sklearn.model_selection import train_test_split
from random import uniform
import pandas as pd
import decimal

#**********************FUNÇÃO DE ATIVAÇÃO***************************
def getFnet(net):
    return (1 / (1 + np.exp(-net)))
    
#**********************DERIVADA DA FUNÇÃO DE ATIVAÇÃO***************
def getDerFnet(fnet):
    return (np.multiply(fnet, (1 - fnet)))

#**********************MODELO INICIAL DA ARQUITETURA****************
class MLP_Arquitetura:
    
    def __init__(self, entrada, oculta, oculta2, saida):
        
        self.entrada = entrada
        self.oculta = oculta
        self.oculta2 = oculta2
        self.saida = saida
        self.hidden_model = self.initHiddenModel()
        self.hidden_model2 = self.initHiddenModel2()
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
    
    
    #**********************INICIAR O MODELO DA CAMADA OCULTA 2********
    def initHiddenModel2(self):
         
        #************PESOS E THETAS PARA A CAMADA OCULTA************
        vetor_oculta2 = np.array([])
        cont = 0
        while(cont < (self.oculta2 * (self.oculta+1))):
            vetor_oculta2 = np.append(vetor_oculta2,uniform(-1,1))
            cont+=1
         
        aux = vetor_oculta2.reshape(self.oculta2, (self.oculta+1))
        return np.asmatrix(aux)
        
    #**********************INICIAR O MODELO DA CAMADA DE SAÍDA******
    def initOutputModel(self):
        
        #************PESOS E THETAS PARA A CAMADA DE SAÍDA**********
        vetor_saida = np.array([])
        cont = 0
        while(cont < (self.saida * (self.oculta2+1))):
            vetor_saida = np.append(vetor_saida,uniform(-1,1))
            cont+=1
         
        aux = vetor_saida.reshape(self.saida, (self.oculta2+1))
        return np.asmatrix(aux)

#**********************FNET = ONDE GUARDO OS RESULTADOS**********************
class Fnet():

    net_hidden = []
    net_hidden2 = []
    fnet_hidden = []
    fnet_hidden2 = []
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
    
    fnet_hidden = np.squeeze(np.asarray(fnet_hidden))
    fnet_hidden = np.append(fnet_hidden,1)
    fnet_hidden = np.asmatrix(fnet_hidden)      
    fnet_hidden = np.transpose(fnet_hidden)
    
    
    #*******************************************
    hidden2  = np.asmatrix(model.hidden_model2)
     
    net_hidden2 = hidden2*fnet_hidden
    
    fnet.net_hidden2 = net_hidden2

    fnet_hidden2 = getFnet(net_hidden2)
    
    fnet.fnet_hidden2 = fnet_hidden2
    
    #*******************************************
    
    fnet_hidden2 = np.squeeze(np.asarray(fnet_hidden2))
    fnet_hidden2 = np.append(fnet_hidden2,1)
    fnet_hidden2 = np.asmatrix(fnet_hidden2)      
    fnet_hidden2 = np.transpose(fnet_hidden2)
    
    #********** SAÍDA DA CAMADA DE SAÍDA**********
    output  = model.output_model
    output = np.asmatrix(output)
    
    net_output = output*fnet_hidden2
    
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
                
                w_o2=np.asmatrix(model.output_model[:,0:model.oculta2])
                
                delta_oculta2 = np.multiply(np.transpose(getDerFnet(resultados.fnet_hidden2)),(delta_saida*w_o2))
                  
                w_o= np.asmatrix(model.hidden_model2[:,0:model.oculta])
                

                delta_oculta = np.multiply(np.transpose(getDerFnet(resultados.fnet_hidden)),(delta_oculta2*w_o))
                
				    #*********************TREINAMENTO***************************
                fnet1 = np.squeeze(np.asarray(resultados.fnet_hidden))
                fnet1 = np.append(fnet1,1)
                fnet1 = np.asmatrix(fnet1)
                
                fnet2 = np.squeeze(np.asarray(resultados.fnet_hidden2))
                fnet2 = np.append(fnet2,1)
                fnet2 = np.asmatrix(fnet2)
                
                # Treinamento da camada de saída
                # w(t+1) = w(t) + eta * dE2_o * i_pj
                
                #************************momentum * modelo***************************
                
                #print(model.output_model)
                
                '''aux = model.output_model.copy()
                
                aux.fill(momentum)
                
                #print(aux)
                
                aux_model = np.multiply(model.output_model,aux)'''
                
                #print(aux_model)
                
                aux_eta = eta*(np.transpose(delta_saida)*fnet2)
                
                #print(aux_eta)
                
                model.output_model = model.output_model+aux_eta
                
                
                aux_eta = eta*(np.transpose(delta_oculta2)*fnet1)
                
                #print(aux_eta)
                
                model.hidden_model2 = model.hidden_model2+aux_eta
            		  		
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
    

#******************************************************************************* 

dataset_output = pd.DataFrame(columns=['cam_ocultas','cam_1','cam_2','parcela_teste','taxa_aprendizado', 'acuracia', 'limite_epocas'])

qtd_entradas = 13
qtd_saidas = 1

v1 = [2,6,13]
v2 = [1,2,3,4,6,13]
eta = [0.2, 0.5]
epoc = [2000, 10000]

for e in epoc:
    
    for x in eta:
        
        for i in v1:
            
            
            for t in v2:    
                
                if ((i == 2 and t == 1) or (i == 6 and t == 3) or (i == 13 and t == 6) or (i == 2 and t == 2) or (i == 6 and t == 4) or (i == 13 and t == 13)):
                    
                    qtd_ocultas = i
                    
                    qtd_ocultas2 = t
                    
                    #********************************NORMALIZAÇÃO MIN-MAX***************************
                    dataset = pd.read_csv('wine.csv')
                    
                    normalizer = sk.preprocessing.MinMaxScaler(feature_range = (0,1))
                    dataset_normalizado = normalizer.fit_transform(dataset)
                    dataset_normalizado = pd.DataFrame(dataset_normalizado, columns= ['Alcohol','Malic.acid','Ash','Acl','Mg','Phenols','Flavanoids','Nonflavanoid.phenols','Proanth','Color.int','Hue','OD','Proline', 'Wine'])
                    
                    #*************EMBARALHANDO E SELECIONANDO PARCELAS DA BASE DE DADOS*************
                    
                    labels = dataset_normalizado['Wine'].values
                    labels = set(labels)
                    
                    dataset_teste = pd.DataFrame(columns=['Alcohol','Malic.acid','Ash','Acl','Mg','Phenols','Flavanoids','Nonflavanoid.phenols','Proanth','Color.int','Hue','OD','Proline', 'Wine'])
                    dataset_treinamento = pd.DataFrame(columns=['Alcohol','Malic.acid','Ash','Acl','Mg','Phenols','Flavanoids','Nonflavanoid.phenols','Proanth','Color.int','Hue','OD','Proline', 'Wine'])
                    
                    
                    for i in labels:
                        d = dataset_normalizado[dataset_normalizado['Wine'] == i]
                        
                        d = sk_utils.shuffle(d)
                        
                        treinamento, teste = train_test_split(d, test_size=0.4)
                        
                        dataset_teste = dataset_teste.append(teste)
                        
                        dataset_treinamento = dataset_treinamento.append(treinamento)
        
                    #*******EMBARALHO O DATASET DE TREINAMENTO PARA NÃO DECORAR A ORDEM*********
                    
                    mlp = MLP_Arquitetura(qtd_entradas,qtd_ocultas, qtd_ocultas2,qtd_saidas)
                    
                    dataset_treinamento=np.asmatrix(dataset_treinamento)
                    dataset_teste=np.asmatrix(dataset_teste)
                    
                    modelo, resultados, cont_epocas = MLP_Backpropagation(mlp, dataset_treinamento, eta = x, limiar = 0.000001, epocas = e)
                    
                    
                    saida_desejada = []
                    print('***************SAÍDA DESEJADA*******************')
                    for linha in range(dataset_teste.A.shape[0]):
                        aux = dataset_teste[linha,mlp.entrada:dataset_teste.A.shape[1]] 
                        saida_desejada = np.append(saida_desejada,aux)
                         
                    #print('************************************************')
                    
                    print(saida_desejada)
                    
                    print('***************SAÍDA TREINADA*******************')
                    
                    saida_treinada = []
                    
                    for linha in range(dataset_teste.A.shape[0]):
                        aux = MLP_Forward(modelo,dataset_teste[linha,0:mlp.entrada],resultados)
                        aux = np.squeeze(np.asarray(aux.fnet_output))
                         
                        
                        ######################ARREDONDAMENTO DA SAÍDA#####################################
                        #exemplo: informo se o número retornado é mais próximo de qual classe normalizada!
                        min_dist = 2.0
                        num = 10.0
                        for i in labels:
                            dif = abs(i - aux)
                            if(dif < min_dist):
                               min_dist = dif
                               num = i
                               
                        saida_treinada = np.append(saida_treinada,num)
                    
                    print(saida_treinada)
                    #************************************ACURÁCIA***********************************
                    somatoria = 0
                    if(len(saida_desejada) == len(saida_treinada)):
                        for linha in range(len(saida_desejada)):
                            if(saida_desejada[linha] == saida_treinada[linha]):
                                somatoria +=1
                    
                    acuracia = (somatoria / len(saida_desejada)) * 100
                    
                    d = decimal.Decimal(acuracia)
                    acuracia = round(d,2)
                    
                    print('Qtd de camadas ocultas: 2')
                    print('Qtd de neurônios na camada oculta 1: '+str(qtd_ocultas))
                    print('Qtd de neurônios na camada oculta 2: '+str(qtd_ocultas2))
                    print('Parcela de teste: '+str(0.4))
                    print('Taxa de aprendizado: '+str(x))
                    print('Acurácia: '+str(acuracia))
                    print('Limite de epocas: '+str(e))

                    df_aux = pd.DataFrame([['2', qtd_ocultas, qtd_ocultas2, 0.4, x, acuracia, e, cont_epocas]], columns=['cam_ocultas','cam_1','cam_2','parcela_teste','taxa_aprendizado', 'acuracia', 'limite_epocas', 'cont_epocas'])
                    dataset_output = pd.concat([dataset_output, df_aux])
                    
                    
dataset_output.to_csv('2OCULTA_40teste.csv')

