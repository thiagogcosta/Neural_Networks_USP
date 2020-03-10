# -*- coding: utf-8 -*-
"""
@Autor: Thiago Aparecido Gonçalves da Costa
@Disciplina: Redes Neurais

@**********************Exercício 1**********************

Descrição: Implementar e treinar o modelo Adaline para reconhecer os símbolos 
A e A invertida​).

Adaline: O processo adaptativo do Adaline consiste em utilizar a função de ativação 
hard limiter (saída +1 ou -1) e minimizar os pesos usando o algoritmo LMS.

Observação 1: No enunciado do exercício não há a descrição da taxa de aprendizado, 
desse modo, para os testes será utilizado o valor 0,5.

Observação 2: Para uma boa representação de um A ou A invertido use matrizes 
quadradas (mesmo número de linhas e colunas).
Por exemplo:
    
+1​ -1 -1 -1 +1
+1​ -1 -1 -1 +1
-1 +1​ ​+1​ +1​ ​-1
-1 +1​ -1 +1​ -1
-1 -1 +1​ -1 -1
"""

import pandas as pd
import numpy as np

#********************VERFICAÇÃO DE SAÍDA**************************
def Verificacao_saida(saida, taxa):
    if(saida < taxa):
        return -1
    else:
        return 1
    
#********************CALCULAR A FUNÇÃO DE ERRO********************
def Erro(t, sout):
    return  t - sout

#********************CALCULAR A SAIDA DO PERCEPTRON***************
def Saida(entradas, pesos, bias):
    cont = 0
    somatorio = 0
    while(cont < len(entradas)):
        somatorio += entradas[cont] * pesos[cont]
        cont+=1

    return (somatorio + bias)

#********************CALCULAR A ATUALIZAÇÃO DE PESOS**************
def Atualizacao_peso(pesos, taxa_aprendizagem, erro, entradas, bias):
    cont = 0
    while(cont < len(entradas)):
        pesos[cont] = pesos[cont] + (taxa_aprendizagem * erro * entradas[cont])
        cont+=1  

    bias = bias + (taxa_aprendizagem * erro)
    return pesos, bias  
    
#********************OBTER ARQUIVO DE TREINO**********************
def getTreino(taxa, bias, epocas):
    #Observação: Colocar o caminho do arquivo de treino
    base_dados = pd.read_csv('C:\\Users\\thiag\\Desktop\\ThiagoApGCosta_exercicio1\\treinamento.csv')
    
    vetor_entrada = []
    cont_pd = 0
    qtd_treinamento = len(base_dados['entrada0'])
    
    while(cont_pd < qtd_treinamento):
        vetor_entrada.append(base_dados['entrada0'][cont_pd])
        vetor_entrada.append(base_dados['entrada1'][cont_pd])
        vetor_entrada.append(base_dados['entrada2'][cont_pd])
        vetor_entrada.append(base_dados['entrada3'][cont_pd])
        vetor_entrada.append(base_dados['entrada4'][cont_pd])
        vetor_entrada.append(base_dados['entrada5'][cont_pd])
        vetor_entrada.append(base_dados['entrada6'][cont_pd])
        vetor_entrada.append(base_dados['entrada7'][cont_pd])
        vetor_entrada.append(base_dados['entrada8'][cont_pd])
        vetor_entrada.append(base_dados['entrada9'][cont_pd])
        vetor_entrada.append(base_dados['entrada10'][cont_pd])
        vetor_entrada.append(base_dados['entrada11'][cont_pd])
        vetor_entrada.append(base_dados['entrada12'][cont_pd])
        vetor_entrada.append(base_dados['entrada13'][cont_pd])
        vetor_entrada.append(base_dados['entrada14'][cont_pd])
        vetor_entrada.append(base_dados['entrada15'][cont_pd])
        vetor_entrada.append(base_dados['entrada16'][cont_pd])
        vetor_entrada.append(base_dados['entrada17'][cont_pd])
        vetor_entrada.append(base_dados['entrada18'][cont_pd])
        vetor_entrada.append(base_dados['entrada19'][cont_pd])
        vetor_entrada.append(base_dados['entrada20'][cont_pd])
        vetor_entrada.append(base_dados['entrada21'][cont_pd])
        vetor_entrada.append(base_dados['entrada22'][cont_pd])
        vetor_entrada.append(base_dados['entrada23'][cont_pd])
        vetor_entrada.append(base_dados['entrada24'][cont_pd])
        vetor_entrada.append(base_dados['entrada25'][cont_pd])
        
        cont_pd +=1
    
    vetor_entrada = np.reshape(vetor_entrada,(qtd_treinamento,26)) 
    
    vetor_pesos = [0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0]
    vetor_saida = base_dados['desejado'].values

    tamanho_vetor = len(vetor_entrada)
    
    cont = 0
    epocas_cont = 0
    cont_nerro = 0
    
    while(epocas_cont<epocas):
        saida_obtida = Verificacao_saida(Saida(vetor_entrada[cont], vetor_pesos, bias), taxa)
        erro = Erro(vetor_saida[cont],saida_obtida)
        if (saida_obtida != vetor_saida[cont]):
            cont_nerro += 1
            vetor_pesos, bias = Atualizacao_peso(vetor_pesos,taxa,erro,vetor_entrada[cont], bias)

        cont+=1
        epocas_cont+=1
        
        if (cont == tamanho_vetor and cont_nerro == 0):
            print("*********TREINADO COM SUCESSO!!!*********")
            print('Vetor de pesos: '+str(vetor_pesos))
            break
        
        if(cont == tamanho_vetor):
            cont = 0
            cont_nerro = 0
    
    print('Treinado em: '+str(epocas_cont)+' épocas')
    print('Bias: '+str(bias))     
    return vetor_pesos, bias 

#********************OBTER ARQUIVO DE TESTE***********************
def getTeste(vetor_pesos, taxa, bias):
    
    #Observação: Colocar o caminho do arquivo de teste
    base_dados = pd.read_csv('C:\\Users\\thiag\\Desktop\\ThiagoApGCosta_exercicio1\\teste.csv')
    
    vetor_entrada = []
    cont_pd = 0
    qtd_testes = len(base_dados['entrada0'])
    
    while(cont_pd < qtd_testes):
        vetor_entrada.append(base_dados['entrada0'][cont_pd])
        vetor_entrada.append(base_dados['entrada1'][cont_pd])
        vetor_entrada.append(base_dados['entrada2'][cont_pd])
        vetor_entrada.append(base_dados['entrada3'][cont_pd])
        vetor_entrada.append(base_dados['entrada4'][cont_pd])
        vetor_entrada.append(base_dados['entrada5'][cont_pd])
        vetor_entrada.append(base_dados['entrada6'][cont_pd])
        vetor_entrada.append(base_dados['entrada7'][cont_pd])
        vetor_entrada.append(base_dados['entrada8'][cont_pd])
        vetor_entrada.append(base_dados['entrada9'][cont_pd])
        vetor_entrada.append(base_dados['entrada10'][cont_pd])
        vetor_entrada.append(base_dados['entrada11'][cont_pd])
        vetor_entrada.append(base_dados['entrada12'][cont_pd])
        vetor_entrada.append(base_dados['entrada13'][cont_pd])
        vetor_entrada.append(base_dados['entrada14'][cont_pd])
        vetor_entrada.append(base_dados['entrada15'][cont_pd])
        vetor_entrada.append(base_dados['entrada16'][cont_pd])
        vetor_entrada.append(base_dados['entrada17'][cont_pd])
        vetor_entrada.append(base_dados['entrada18'][cont_pd])
        vetor_entrada.append(base_dados['entrada19'][cont_pd])
        vetor_entrada.append(base_dados['entrada20'][cont_pd])
        vetor_entrada.append(base_dados['entrada21'][cont_pd])
        vetor_entrada.append(base_dados['entrada22'][cont_pd])
        vetor_entrada.append(base_dados['entrada23'][cont_pd])
        vetor_entrada.append(base_dados['entrada24'][cont_pd])
        vetor_entrada.append(base_dados['entrada25'][cont_pd])
        
        cont_pd +=1
    
    #modificar para testar outros 
    vetor_entrada = np.reshape(vetor_entrada,(qtd_testes,26))
    
    cont = 0
    
    while(cont<qtd_testes):
        saida_obtida = Verificacao_saida(Saida(vetor_entrada[cont], vetor_pesos, bias), taxa)
        print(saida_obtida)
        cont+=1
        
#****************************MAIN***************************

def main(taxa, bias, epocas):
    
    vetor_pesos, bias = getTreino(taxa, bias, epocas)
    getTeste(vetor_pesos, taxa, bias)
        
if __name__ == "__main__":
    taxa = 0.5
    bias = 0.5
    epocas = 100000
    main(taxa, bias, epocas)
    
    
    
