from Instance_semantic_segmentation_model import unet_multiclass
from data_multiclass import data_multiclass
import tensorflow as tf
import datetime
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import MeanIoU
import random
from tensorflow.keras.models import load_model
import cv2



X_train, Y_train_cat, Y_train, X_test, Y_test_cat,Y_test, n_classes, class_weights= data_multiclass()
IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH  = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]
model =unet_multiclass(n_classes, IMG_HEIGHT, IMG_WIDTH,IMG_CHANNELS)
model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
model.load_weights('multiclass_model_512.h5')


_, acc = model.evaluate(X_test, Y_test_cat)
print("Accuracy is = ", (acc * 100.0), "%")

y_pred=model.predict(X_test)
y_pred=y_pred.reshape((X_test.shape[0],128,128,n_classes))
y_pred_argmax=np.argmax(y_pred, axis=3)
print(y_pred_argmax.shape)

##################################################

#Using built in keras function
#n_classes = 6
#IOU_keras = MeanIoU(num_classes=n_classes)  
#IOU_keras.update_state(Y_test[:,:,:,0], y_pred_argmax)
#print("Mean IoU =", IOU_keras.result().numpy())
#
#
##To calculate I0U for each class...
#values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
#print(values)
#class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[0,4]+ values[0,5] + values[0,6]+ values[1,0]+ values[2,0]+ values[3,0] + values[4,0]+ values[5,0] + #values[6,0])
#class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[1,4]+ values[1,5] + values[1,6]+ values[0,1]+ values[2,1]+ values[3,1] + values[4,1]+ values[5,1] + #values[6,1])
#class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[2,4]+ values[2,5] + values[2,6]+ values[0,2]+ values[1,2]+ values[3,2] + values[4,2]+ values[5,2] + #values[6,2])
#class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[3,4]+ values[3,5] + values[3,6]+ values[0,3]+ values[1,3]+ values[2,3] + values[4,3]+ values[5,3] + #values[6,3])
#class5_IoU = values[4,4]/(values[4,4] + values[4,0] + values[4,1] + values[4,2] + values[4,3]+ values[4,5] + values[4,6]+ values[0,4]+ values[1,4]+ values[2,4] + values[3,4]+ values[5,4] + #values[6,4])
#class6_IoU = values[5,5]/(values[5,5] + values[5,0] + values[5,1] + values[5,2] + values[5,3]+ values[5,4] + values[5,6]+ values[0,5]+ values[1,5]+ values[2,5] + values[3,5]+ values[4,5] + #values[6,5])
##class7_IoU = values[6,6]/(values[6,6] + values[6,0] + values[6,1] + values[6,2] + values[6,3]+ values[6,4] + values[6,5]+ values[0,6]+ values[1,6]+ values[2,6] + values[3,6]+ values[4,6] + values[5,6])
#print("IoU for class1 is: ", class1_IoU)
#print("IoU for class2 is: ", class2_IoU)
#print("IoU for class3 is: ", class3_IoU)
#print("IoU for class4 is: ", class4_IoU)
#print("IoU for class5 is: ", class5_IoU)
#print("IoU for class6 is: ", class6_IoU)
##print("IoU for class7 is: ", class7_IoU)



test_img_number = random.randint(0, len(X_test)-1)
test_img = X_test[test_img_number]
test_img_input = test_img[None, :,:,:]
ground_truth=Y_test[test_img_number]
prediction = (model.predict(test_img_input))
prediction= prediction.reshape((1,128,128,n_classes))
 
predicted_img=np.argmax(prediction, axis=3)[0,:,:]

plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img, cmap= 'viridis_r')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='jet')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(predicted_img, cmap='jet')   
plt.show()
plt.close()



#print(Y_train_cat.shape)
#
#
#
#X_test = X_test[1][:,:,None]
#print(X_test.shape)
#
#X_test =normalize(X_test, axis =1)
#print(X_test.shape)
#prediction = model.predict(X_test)
#print(prediction.shape)
#y_pred = np.argmax(prediction,axis = 4)[0,:,:]
#print(y_pred.max())
#plt.imshow(y_pred)
#plt.show()






