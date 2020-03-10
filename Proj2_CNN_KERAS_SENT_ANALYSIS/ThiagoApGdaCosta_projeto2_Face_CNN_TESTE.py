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
from os.path import join
import glob
import pickle
import time
import sys


#************DIMENSÕES DA IMAGEM*****************
linhas, colunas = 224, 224

#*********SELEÇÃO DO MODELO TREINADO*************
modelo = pickle.load(open('face_model_arq1_16Batchs_25Epocas.sav', 'rb'))

#*************LEITURA DAS FOTOS*************
x_teste = []
y_teste = []

imagens = ('angry','fear','happy1','happy2','happy3','neutral','surprise')
tipos_imagem = ('*.jpeg', '*.jpg')

for img in imagens:
    for ti in tipos_imagem:
        for filename in glob.glob('FACES/'+str(img)+str(ti)):

            image = 0
            rotulo = 0

            try:
                 image = Image.open(filename)

                 if image is not None:

                     #*************REDIMENSIONAMENTO DAS IMAGENS*************
                     image = image.resize((linhas, colunas), Image.ANTIALIAS).convert('L')

                     image = np.array(image)

                     #***************RÓTULOS*************
                     if(img == 'angry'):
                         rotulo = 0
                     elif(img == 'disgust'):
                         rotulo = 1
                     elif(img == 'fear'):
                         rotulo = 2
                     elif(img == 'happy1' or img == 'happy2' or img == 'happy3'):
                         rotulo = 3
                     elif(img == 'negative'):
                         rotulo = 4
                     elif(img == 'neutral'):
                         rotulo = 5
                     else:
                         rotulo = 6

                     x_teste.append(image)
                     y_teste.append(rotulo)


            except IOError:
                print("Impossível abrir a imagem!")
                sys.exit(1)


#**********NORMALIZAÇÃO DO TESTE************
x_teste = np.asarray(x_teste)

x_teste = x_teste.reshape(len(imagens), linhas, colunas, 1)
x_teste = x_teste.astype('float32')
x_teste /= 255

#**********INFERÊNCIA DO TESTE************
for i in range(len(imagens)):
    imagem = x_teste[i]

    imagem = (np.expand_dims(imagem,0))

    predictions_single = modelo.predict(imagem)

    resultado = np.argmax(predictions_single)

    print('*******imagem:'+str(i)+'*******')
    print('Saída desejada: ', y_teste[i])
    print('Saída obtida: ', resultado)
