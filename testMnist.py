# -*- coding: utf-8 -*-
from keras import backend as K
from keras.preprocessing.image import load_img,img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
import numpy as np

# Pour utiliser tensorflow
K.set_image_data_format('channels_last')

modele1 = load_model('modele_TP2.h5')
modele2 = load_model('modele_TP2_CNN.h5')
modele3 = load_model('modele_TP2_net.h5')

while True:
    img = load_img('img.png',color_mode="grayscale",target_size=(28,28))
    img = img_to_array(img)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    
    prediction1 = modele1.predict(img)
    print("Reseau du TP : ",np.argmax(prediction1)," ",prediction1[0][np.argmax(prediction1)])
    
    prediction2 = modele2.predict(img)
    print("Reseau du TP avec CNN : ",np.argmax(prediction2)," ",prediction2[0][np.argmax(prediction2)])
    prediction3 = modele3.predict(img
                                  )
    print("Reseau LeNet : ",np.argmax(prediction3)," ",prediction3[0][np.argmax(prediction3)])
    input("Appuyer sur Entree pour recharger l'image")
    

