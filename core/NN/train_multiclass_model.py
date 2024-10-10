from sklearn.utils import class_weight
from tensorflow.python.framework.constant_op import constant
from core.NN.Instance_semantic_segmentation_model import unet_multiclass
from core.NN.data_multiclass  import data_multiclass
import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow.keras as keras
import datetime
import numpy as np
from itertools import product
from focal_loss import SparseCategoricalFocalLoss




def train_multiclass():
    X_train, Y_train_cat,Y_train, X_test, Y_test_cat, Y_test, n_classes, class_weights= data_multiclass()
    IMG_HEIGHT = X_train.shape[1]
    IMG_WIDTH  = X_train.shape[2]
    IMG_CHANNELS = X_train.shape[3]
    model =unet_multiclass(n_classes, IMG_HEIGHT, IMG_WIDTH,IMG_CHANNELS)
    sample_weights = np.zeros((X_train.shape[0],IMG_HEIGHT*IMG_WIDTH,n_classes))
    for i in range(n_classes):
        sample_weights[:,:,i] += class_weights[i]


    print(sample_weights.shape)
    



    #class_weight_dict= dict(zip(range(n_classes),class_weights))
    model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    checkpointer = tf.keras.callbacks.ModelCheckpoint('multiclass_model_512_test.h5',verbose=1, save_best_only=True,sample_weight_mode='temporal')
    log_director = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=6,monitor='val_loss'),    
    tf.keras.callbacks.TensorBoard(
    log_dir=log_director, histogram_freq=1, write_graph=True,
    write_images=True, update_freq='epoch', profile_batch=0,
    embeddings_freq=1),checkpointer]
    results = model.fit(X_train,Y_train_cat,batch_size =10,epochs=100,callbacks=callbacks, validation_data=(X_test, Y_test_cat), shuffle = False, sample_weight = np.squeeze(np.sum(sample_weights, axis=-1)))
    return results


results = train_multiclass()
print(results)




