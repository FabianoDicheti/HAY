#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 22:24:27 2022

@author: fabianodicheti
"""

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from copy import copy as cp
imagem = Image.open('ateste.png')

matriz_imagem = np.asarray(imagem)


matriz_imagem.shape

plt.imshow(matriz_imagem[:,:,2])

imagem2 = Image.open('ateste 2.png')

matriz_imagem2 = np.asarray(imagem2)



mat1 = matriz_imagem2[:,:,0]
mat2 = matriz_imagem2[:,:,1]
mat3 = matriz_imagem2[:,:,2]


plt.imshow(mat3, cmap='gray')


junto = np.dstack((mat1, mat2, mat3)) ### plot em RGB sequencia

plt.imshow(junto)


normR = [b*2 for b in mat1]
normG = [r*2 for r in mat2]
normB = [iv*2 for iv in mat3]

tog = np.dstack((normR, normG, normB))

tog2 = cv2.resize(tog, (400,400), interpolation = cv2.INTER_AREA)
plt.imshow(tog2, cmap='gray')

tog3 = cp(tog2)

maximo = 255
minimo = 0
limiar = round(tog2.max()*0.5,0).astype('int')
conversao_inicial = np.where((tog2 <= limiar), tog2, maximo)
conversao_final = np.where((conversao_inicial> limiar),conversao_inicial, minimo)


image_eroded = cv2.dilate(src=conversao_final, kernel=np.ones((4, 4)), iterations=2)


plt.imshow(image_eroded, cmap='gray')
