# -*- coding: utf-8 -*-
"""
@Autor: Thiago Aparecido Gonçalves da Costa
@Disciplina: Redes Neurais

**********************EXERCÍCIO 4**********************

DATASET = MNIST
TIPO DE REDE NEURAL: CNN

VALIDAÇÃO: CLASSIFICAÇÃO DE NÚMEROS ESCRITOS A MÃO
"""

#***************************BIBLIOTECAS*******************************
import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras import backend as K
import numpy as np
from PIL import Image
import sys
import cv2

#**********PRÉ-PROCESSAMENTO DOS DADOS******
batch_size = 512
num_classes = 10
epocas = 10

#************DIMENSÕES DA IMAGEM*************
linhas, colunas = 28, 28

#**********CARREGANDO A BASE DE DADOS*********
dados = mnist.load_data()

#******DIVISÃO EM TREINAMENTO E TESTE*********
(x_treinamento, y_treinamento), (x_teste, y_teste) = dados

#***REDIMENSIONAMENTO DAS BASES DE TREINAMENTO E TESTE*********
x_treinamento = x_treinamento.reshape(x_treinamento.shape[0], linhas, colunas, 1)
x_teste = x_teste.reshape(x_teste.shape[0], linhas, colunas, 1)
input_shape = (linhas, colunas, 1)

#*************NORMALIZAÇÃO*************
x_treinamento = x_treinamento.astype('float32')
x_teste = x_teste.astype('float32')
x_treinamento /= 255
x_teste /= 255

print('CONJUNTO DE DADOS:', x_treinamento.shape)
print('TREINAMENTO: ', x_treinamento.shape[0])
print('TESTE: ', x_teste.shape[0])

#*******************************GPU*******************************
#Criando sessão para que o tensorflow + cuda rode a CNN na GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)
#******************************************************************

y_treinamento = keras.utils.to_categorical(y_treinamento, num_classes)
y_teste = keras.utils.to_categorical(y_teste, num_classes)

#*******************************CNN********************************
model = Sequential()

#*************************CAMADAS CONVOLUCIONAIS*******************
model.add(Conv2D(128, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(128, (5, 5), activation='relu'))
#*************************MaxPooling2D*******************
model.add(MaxPooling2D(pool_size=(2, 2)))
#*************************CAMADAS CONVOLUCIONAIS*******************
model.add(Conv2D(256, (2, 2), activation='relu'))
model.add(Conv2D(256, (2, 2), activation='relu'))
#*************************MaxPooling2D*******************
model.add(MaxPooling2D(pool_size=(2, 2)))
#*************************CAMADAS CONVOLUCIONAIS*******************
model.add(Conv2D(512, (2, 2), activation='relu'))
model.add(Conv2D(512, (2, 2), activation='relu'))
#*************************MaxPooling2D*******************
model.add(MaxPooling2D(pool_size=(2, 2)))
#*************************Dropout*******************
model.add(Dropout(0.5))
#*************************Flatten*******************
model.add(Flatten())

#****************CAMADAS TOTALMENTE CONECTADAS********************
model.add(Dense(2048, activation='relu'))
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))

#***********************CAMADA DE SAÍDA****************************
model.add(Dense(num_classes, activation='softmax'))

#***********************TREINAMENTO*********************************
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=Adam(lr=0.0001),
              metrics=['accuracy'])

model.fit(x_treinamento, y_treinamento,
          batch_size=batch_size,
          epochs=epocas,
          verbose=1,
          validation_data=(x_teste, y_teste))

#***********************TESTE*********************************
score = model.evaluate(x_teste, y_teste, verbose=0)

print('% DE PERDA:', score[0]*100)
print('% DE ACURÁCIA:', score[1]*100)

#*************LEITURA DAS FOTOS*************
#FOTOS SEM RUÍDO

digitos_teste = [4,5,8,9]

for i in digitos_teste:

    try:
        img = cv2.imread('fotos_digitos/'+str(i)+'.jpeg',0)
        kernel = np.ones((5,5), np.uint8)
        #APLICAÇÃO DE EROSÃO E DILATAÇÃO NA IMAGEM PARA RETIRAR RUÍDOS E REALÇAR AS CARACTERÍSTICAS PRINCIPAIS
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        #INVERSÃO DAS CORES, VISTO QUE O NÚMERO DEVE ESTAR COM COR CLARA
        img = cv2.bitwise_not(opening)
        digito = Image.fromarray(img)

    except IOError:
        print("Unable to load image")
        sys.exit(1)

    #**********VISUALIZAÇÕES DAS MODIFICAÇÕES FEITAS PELO OPENCV************
    digito.show()

    #**********NORMALIZAÇÃO DO TESTE************
    digito = digito.resize((28, 28), Image.ANTIALIAS)

    digito = np.array(digito)

    digito = digito.reshape(1, linhas, colunas, 1)

    digito = digito.astype('float32')
    digito /= 255

    #**********INFERÊNCIA DO TESTE************
    predictions_single = model.predict(digito)

    resultado = np.argmax(predictions_single[0])

    print('SAÍDA DESEJADA', i)
    print('SAÍDA', resultado)
