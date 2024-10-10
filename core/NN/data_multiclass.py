
import os
import glob
from tkinter import Image
import cv2
import numpy as np


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.utils import class_weight



def data_multiclass():
    
    TRAIN_PATH1 = 'training_512/1_color/'
    TRAIN_PATH2 = 'training_512/2_color/'
    TRAIN_PATH3 = 'training_512/8_color/'
    TRAIN_PATH4 = 'training_512/16_color/'
    TRAIN_PATH5 = 'training_512/29_color/'
    TRAIN_PATH6 = 'training_512/1_Dinolite/'
    TRAIN_PATH7 = 'training_512/10_Nikon/'
    TRAIN_PATH8 = 'training_512/19_color/'
    
    print('getting data')

    train_img =[]
    for i in range(3):
        for j in range(6):
            img = cv2.imread(TRAIN_PATH1+str(i)+'_'+str(j)+'_1_normal.png')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            train_img.append(img)
    
    for i in range(3):
        for j in range(6):
            img = cv2.imread(TRAIN_PATH2+str(i)+'_'+str(j)+'_2_normal.png')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            train_img.append(img)

    for i in range(3):
        for j in range(6):
            img = cv2.imread(TRAIN_PATH3+str(i)+'_'+str(j)+'_8_normal.png')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            train_img.append(img)
    
    for i in range(3):
        for j in range(6):
            img = cv2.imread(TRAIN_PATH4+str(i)+'_'+str(j)+'_16_normal.png')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            train_img.append(img)

    for i in range(2):
        for j in range(7):
            img = cv2.imread(TRAIN_PATH5+str(i)+'_'+str(j)+'_29_normal.png')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            train_img.append(img)

    for i in range(6):
        for j in range(7):
            img = cv2.imread(TRAIN_PATH6+str(i)+'_'+str(j)+'_1_normal.png')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            train_img.append(img)

    for i in range(10):
        for j in range(10):
            img = cv2.imread(TRAIN_PATH7+str(i)+'_'+str(j)+'_10_normal.png')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            train_img.append(img)
    
    for i in range(3):
        for j in range(6):
            img = cv2.imread(TRAIN_PATH8+str(i)+'_'+str(j)+'_19_normal.png')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            train_img.append(img)

    train_mask = []
    for i in range(3):
        for j in range(6):
            img = cv2.imread(TRAIN_PATH1+str(i)+'_'+str(j)+'_1_diameter.png',0)
            train_mask.append(img)
    
    for i in range(3):
        for j in range(6):
            img = cv2.imread(TRAIN_PATH2+str(i)+'_'+str(j)+'_2_diameter.png',0)
            train_mask.append(img)

    for i in range(3):
        for j in range(6):
            img = cv2.imread(TRAIN_PATH3+str(i)+'_'+str(j)+'_8_diameter.png',0)
            train_mask.append(img)

    for i in range(3):
        for j in range(6):
            img = cv2.imread(TRAIN_PATH4+str(i)+'_'+str(j)+'_16_diameter.png',0)   
            train_mask.append(img)
    for i in range(2):
        for j in range(7):
            img = cv2.imread(TRAIN_PATH5+str(i)+'_'+str(j)+'_29_diameter.png',0)   
            train_mask.append(img)
    for i in range(6):
        for j in range(7):
            img = cv2.imread(TRAIN_PATH6+str(i)+'_'+str(j)+'_1_diameter.png',0)   
            train_mask.append(img)
    for i in range(10):
        for j in range(10):
            img = cv2.imread(TRAIN_PATH7+str(i)+'_'+str(j)+'_10_diameter.png',0)   
            train_mask.append(img)
    for i in range(3):
        for j in range(6):
            img = cv2.imread(TRAIN_PATH8+str(i)+'_'+str(j)+'_19_diameter.png',0)   
            train_mask.append(img)
    
    

    train_images= np.array(train_img)
    train_masks= np.array(train_mask)
    
    
   

    labelencoder = LabelEncoder()
    n, h, w = train_masks.shape
    
    train_masks_reshaped = train_masks.reshape(-1,1)[:,0]
    
   
    train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
    train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)

    n_classes = len(np.unique(train_masks_encoded_original_shape))
    

    #################################################
    

    train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)

    #Create a subset of data for quick testing
    #Picking 10% for testing and remaining for training
    x_train, x_test, y_train, y_test = train_test_split(train_images, train_masks_input, test_size = 0.10, random_state =5)


    #Further split training data t a smaller subset for quick testing of models
    

    

    from tensorflow.keras.utils import to_categorical
    train_masks_cat = to_categorical(y_train, num_classes=n_classes)
    y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1]*y_train.shape[2], n_classes))



    test_masks_cat = to_categorical(y_test, num_classes=n_classes)
    y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1]*y_test.shape[2], n_classes))

   
    class_weights = class_weight.compute_class_weight(class_weight='balanced',classes = np.unique(train_masks_reshaped_encoded),y = train_masks_reshaped_encoded)
   
    #smote = BorderlineSMOTE()
    #X_sm, Y_sm = smote.fit_resample(x_train,y_train_cat)



    ###############################################################







    return x_train, y_train_cat, y_train, x_test, y_test_cat, y_test, n_classes, class_weights



x_train, y_train_cat, y_train, x_test, y_test_cat, y_test, n_classes, class_weights= data_multiclass()
print(x_train.shape)
