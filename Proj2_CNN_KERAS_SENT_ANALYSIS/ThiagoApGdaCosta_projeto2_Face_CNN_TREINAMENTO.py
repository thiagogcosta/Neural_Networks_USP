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
import glob
from os.path import join
import pickle
import time
import sys


#************DIMENSÕES DA IMAGEM*****************
linhas, colunas = 224, 224

inicio = time.time()

#**********PRÉ-PROCESSAMENTO DOS DADOS******
batch_size = 16
num_classes = 8
epocas = 200

#**********CARREGANDO A BASE DE DADOS*********
x_treinamento = []
y_treinamento = []
x_teste = []
y_teste = []

tipos_imagem = ('*.jpeg', '*.png', '*.jpg')
tipos_dados = ('train','val')
pastas = ('angry', 'disgust', 'fear', 'happy','negative','neutral', 'sad', 'surprise')

for td in tipos_dados:
    for p in pastas:
        for ti in tipos_imagem:
            for filename in glob.glob('QIDER/'+str(td)+'/'+str(p)+'/'+str(ti)):

                image = 0
                rotulo = 0

                try:
                     image = Image.open(filename)

                     if image is not None:

                         #*************REDIMENSIONAMENTO DAS IMAGENS*************
                         image = image.resize((linhas, colunas), Image.ANTIALIAS)

                         image = np.array(image)

                         #***************RÓTULOS*************
                         if(p == 'angry'):
                             rotulo = 0
                         elif(p == 'disgust'):
                             rotulo = 1
                         elif(p == 'fear'):
                             rotulo = 2
                         elif(p == 'happy'):
                             rotulo = 3
                         elif(p == 'negative'):
                             rotulo = 4
                         elif(p == 'neutral'):
                             rotulo = 5
                         elif(p == 'sad'):
                             rotulo = 6
                         else:
                             rotulo = 7
                         if(td == 'train'):
                             x_treinamento.append(image)
                             y_treinamento.append(rotulo)
                         else:
                             x_teste.append(image)
                             y_teste.append(rotulo)
                except IOError:
                    print("Impossível abrir a imagem!")
                    sys.exit(1)


#***REDIMENSIONAMENTO DAS BASES DE TREINAMENTO E TESTE*********
print('CONJUNTO DE DADOS: ', len(x_treinamento) + len(x_teste))
print('TREINAMENTO: ', len(x_treinamento))
print('TESTE: ', len(x_teste))


x_treinamento = np.asarray(x_treinamento)
y_treinamento = np.asarray(y_treinamento)
x_teste = np.asarray(x_teste)
y_teste = np.asarray(y_teste)

x_treinamento = x_treinamento.reshape(len(x_treinamento), linhas, colunas, 1)
x_teste = x_teste.reshape(len(x_teste), linhas, colunas, 1)
input_shape = (linhas, colunas, 1)

#*************NORMALIZAÇÃO*************
x_treinamento = x_treinamento.astype('float32')
x_teste = x_teste.astype('float32')
x_treinamento /= 255
x_teste /= 255

#*******************************GPU*******************************
#Criando sessão para que o tensorflow + cuda rode a CNN na GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth = False
session = tf.Session(config=config)
K.set_session(session)
#******************************************************************

y_treinamento = keras.utils.to_categorical(y_treinamento, num_classes)
y_teste = keras.utils.to_categorical(y_teste, num_classes)

#*******************************CNN********************************
print('CNN INICIALIZADA!')
model = Sequential()

#*************************CAMADAS CONVOLUCIONAIS*******************
model.add(Conv2D(32, (5, 5),activation='relu',input_shape=input_shape))
model.add(Conv2D(32, (5, 5), activation='relu'))
model.add(Conv2D(32, (5, 5), activation='relu'))
#*************************MaxPooling2D*******************
model.add(MaxPooling2D(pool_size=(2, 2)))

#*************************CAMADAS CONVOLUCIONAIS*******************
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
#*************************MaxPooling2D*******************
model.add(MaxPooling2D(pool_size=(2, 2)))

#*************************CAMADAS CONVOLUCIONAIS*******************
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
#*************************MaxPooling2D*******************
model.add(MaxPooling2D(pool_size=(2, 2)))

#*************************Dropout*******************
model.add(Dropout(0.5))
#*************************Flatten*******************
model.add(Flatten())

#****************CAMADAS TOTALMENTE CONECTADAS********************
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

#***********************CAMADA DE SAÍDA****************************
model.add(Dense(num_classes, activation='softmax'))


#***********************TREINAMENTO*********************************
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])

model.fit(x_treinamento, y_treinamento,
          batch_size=batch_size,
          epochs=epocas,
          verbose=1,
          validation_data=(x_teste, y_teste))

#***********************TESTE*********************************
score = model.evaluate(x_teste, y_teste, verbose=0)

print('PERDA: ', score[0])
print('ACURÁCIA: ', score[1])

pickle.dump(model, open('face_model_arq1_16Batchs_200EpocasCERTO.sav', 'wb'))
print('CNN FINALIZADA!')
fim = time.time()
print('TEMPO TOTAL: ',fim - inicio)
