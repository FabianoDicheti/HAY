# Convlolucionar = converter a imagem em matriz
# ReLU = Retificador Linear Unitario / aplicar um retificador "rectifier" para
#                               remover a linearidade na hora de treinar a rede
# Pooling = treinar distorçōes na imagem
# Flattening  = converter a matriz em vetor
# Full Connection = inserir neuronios entre a camada de entrada "Flatt" e a
#                                                            saída "classe"
# 0 = 100% black / 255 = 100% white
# imagens preto e branco sao convertidas em arrays 2d
# imagens coloridas sao convertidas em arrays 3d (RGB) (ou seja cada pixel tem
#                                   3 valores atribuidos, 1 para R, 1 G e 1 B)
# filtro (filter) = feature detector (detector de caracteristicas)
# no final da rede, se usa uma funcao "softmax", para a soma dos resultados
#                                                        ser igual a 1 (100%)
# ao invez de utilizar o MSE para medir o desempenho, na rede neural Conv. é
#                   melhor usar a funcao de entropia cruzada (cross-entropy)
# para imagens a entropia cruzada funciona pelhor pq usa uma escala de
#   logaritmo, ou seja, mais sensivel à melhorias (na back-propagation)


# 1 - Pré processamento

# 2 - Arquitetura da CNN
# 2.1 - convolução
# 2.2 - Pooling
# 2.3 - Flattening
# 2.4 - Full Connection
# 2.5 - Camada de Output

# 3 - Treinar a rede
# 3.1 - Compilar
# 3.2 - Havaliar

# 4 - Deploy


from pickletools import optimize
from warnings import filters
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2, os

### RE ESCALAR PELO DATASET E NAO PELO ARQUIVO... MIN E MAX GERAL.
def cnn1(ref, neurons): #neurons =64
    # pré processamento dos dados de treino
    datagen_treino = ImageDataGenerator(
                        rescale=1./255,
                        shear_range=0.2,
                        zoom_range=0.2,
                        horizontal_flip=True)
    
    training_set = datagen_treino.flow_from_directory(
                                'dataset/training_set',
                                target_size=(50, 50),
                                batch_size=32,
                                class_mode='binary')
    
    # pré processamento dos dados de teste
    datagen_teste = ImageDataGenerator(rescale=1./255)
    
    test_set = datagen_teste.flow_from_directory(
                                'dataset/test_set',
                                target_size=(50, 50),
                                batch_size=32,
                                class_mode='binary')
    
    # inicializar
    cnn = tf.keras.models.Sequential()
    
    #cnn.add(tf.keras.layers.RandomFlip(mode="horizontal_and_vertical"))
    #cnn.add(tf.keras.layers.RandomRotation(0.2))
    
    
    ###### ADD LAYER BATCH NORMALIZATION NO PRIMEIRO DO LAYER DE ENTRADA
    cnn.add(tf.keras.layers.BatchNormalization())
    
    # convolução KERNEL SIZE = 3
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu',
            input_shape=[50, 50, 3]))
    
    ### ADD LAYER BATCH NORMALIZATION DEPOIS DO LAYER DE CONVOLUCAO
    cnn.add(tf.keras.layers.BatchNormalization())
    
    # Pooling
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=3))
    
    # adicionar mais uma camada
    cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
    
    ### ADD LAYER BATCH NORMALIZATION DEPOIS DO LAYER DE CONVOLUCAO
    cnn.add(tf.keras.layers.BatchNormalization())
    
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    
    # Flattening - achatamento, converter em um vetor
    cnn.add(tf.keras.layers.Flatten())
    
    # Full Conection (camada de neuronios ocultos)  ### DROPOUT 0.2
    cnn.add(tf.keras.layers.Dense(units=neurons, activation='relu'))
    #cnn.add(tf.keras.layers.Dropout(rate=0.1))
    
    #cnn.add(tf.keras.layers.Dense(units=neurons, activation='relu'))
    
    # camada de saída (output layer) ATIVACAO SOFTMAX, softmax nao funciona, classifica tudo como soja
    cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    
    # treino da rede
    # compilar a rede 
    cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # treinar
    cnn.fit(x=training_set, validation_data=test_set, epochs=150)
    
    cnn.save("peso/"+ref+"/")
    
    resultados2 = []
    milho2 = 0
    soja2 = 0
    
    
    for alvo in os.listdir("dataset/predizer/feliz/"):
        nome = "dataset/predizer/soja/"+alvo
        imagem_teste = image.load_img(nome, target_size=(50, 50))
        imagem_teste = image.img_to_array(imagem_teste)
        imagem_teste = np.expand_dims(imagem_teste, axis=0)
        resultado = cnn.predict(imagem_teste/255.0)
        training_set.class_indices
        resultados2.append(resultado)
    
    contagem = 0
    for i in range(len(resultados2)):
        a = resultados2[i][0][0]
        if a > 0.5:
            contagem+=1
            
    print('soja: ...') 
    print(contagem/len(resultados2))  
    print(' ...') 
    soja = contagem/len(resultados2)
    
    resultados2 = []
    milho2 = 0
    soja2 = 0
    
    
    for alvo in os.listdir("dataset/predizer/triste/"):
        nome = "dataset/predizer/milho/"+alvo
        imagem_teste = image.load_img(nome, target_size=(50, 50))
        imagem_teste = image.img_to_array(imagem_teste)
        imagem_teste = np.expand_dims(imagem_teste, axis=0)
        resultado = cnn.predict(imagem_teste/255.0)
        training_set.class_indices
        resultados2.append(resultado)
    
    contagem = 0
    for i in range(len(resultados2)):
        a = resultados2[i][0][0]
        if a < 0.5:
            contagem+=1
    



