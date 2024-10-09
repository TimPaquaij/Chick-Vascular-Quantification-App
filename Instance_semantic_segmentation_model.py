import tensorflow as tf
import tensorflow_addons as tfa



def unet_multiclass(n_classes,IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS):
    
    inputs = tf.keras.layers.Input([IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS])
    s = (inputs/255)

    #contraction path
    c1 = tf.keras.layers.Conv2D(16,(3,3),activation='relu',kernel_initializer='he_normal',padding='same',)(s) # 2D Convolution layer with kernel size (3,3)
    c1 = tfa.layers.GroupNormalization(groups= 8, axis = 3)(c1)                                               # Groupnormalisation
    c1 = tf.keras.layers.Conv2D(16,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c1) # 2D Convolution layer with kernel size (3,3)
    c1 = tfa.layers.GroupNormalization(groups= 8, axis = 3)(c1)                                               # Groupnormalisation
    p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)                                                              #Max Pooling

    c2 = tf.keras.layers.Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(p1) 
    c2 = tfa.layers.GroupNormalization(groups= 8, axis = 3)(c2)                                               
    c2 = tf.keras.layers.Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c2) 
    c2 = tfa.layers.GroupNormalization(groups= 8, axis = 3)(c2)                                               
    p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)                                                              

    c3 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(p2)                                                              
    c3 = tfa.layers.GroupNormalization(groups= 8, axis = 3)(c3)                                                            
    c3 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c3)                                                              
    c3 = tfa.layers.GroupNormalization(groups= 8, axis = 3)(c3)                                                            
    p3 = tf.keras.layers.MaxPooling2D((2,2))(c3)                                                               


    c4 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(p3)
    c4 = tfa.layers.GroupNormalization(groups= 8, axis = 3)(c4)
    c4 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c4)
    c4 = tfa.layers.GroupNormalization(groups= 8, axis = 3)(c4)
    p4 = tf.keras.layers.MaxPooling2D((2,2))(c4) 

    c5 = tf.keras.layers.Conv2D(256,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(p4)
    c5 = tfa.layers.GroupNormalization(groups= 8, axis = 3)(c5)
    c5 = tf.keras.layers.Conv2D(256,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c5)
    c5 = tfa.layers.GroupNormalization(groups= 8, axis = 3)(c5)

    #Expansive Path 
    u6 = tf.keras.layers.Conv2DTranspose(128,(2,2),strides =(2,2),padding='same')(c5)                            # 2D Transpose Convolution kernel size (2,2)
    u6 = tfa.layers.GroupNormalization(groups= 8, axis = 3)(u6)
    u6 = tf.keras.layers.concatenate([u6,c4])                                                                      # Concrate to combine decoder side with encoder side
    c6 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(u6)
    c6 = tfa.layers.GroupNormalization(groups= 8, axis = 3)(c6)
    c6 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c6)
    c6 = tfa.layers.GroupNormalization(groups= 8, axis = 3)(c6)

    u7 = tf.keras.layers.Conv2DTranspose(64,(2,2),strides =(2,2),padding='same')(c6)
    u7 = tfa.layers.GroupNormalization(groups= 8, axis = 3)(u7)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(u7)
    c7 = tfa.layers.GroupNormalization(groups= 8, axis = 3)(c7)
    c7= tf.keras.layers.Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c7)
    c7 = tfa.layers.GroupNormalization(groups= 8, axis = 3)(c7)

    u8 = tf.keras.layers.Conv2DTranspose(32,(2,2),strides =(2,2),padding='same')(c7)
    u8 = tfa.layers.GroupNormalization(groups= 8, axis = 3)(u8)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(u8)
    c8 = tfa.layers.GroupNormalization(groups= 8, axis = 3)(c8)
    c8= tf.keras.layers.Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c8)
    c8 = tfa.layers.GroupNormalization(groups= 8, axis = 3)(c8)

    u9 = tf.keras.layers.Conv2DTranspose(16,(2,2),strides =(2,2),padding='same')(c8)
    u9 = tfa.layers.GroupNormalization(groups= 8, axis = 3)(u9)
    u9 = tf.keras.layers.concatenate([u9, c1],axis=3)
    c9 = tf.keras.layers.Conv2D(16,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(u9)
    c9 = tfa.layers.GroupNormalization(groups= 8, axis = 3)(c9)
    c9= tf.keras.layers.Conv2D(16,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c9)
    c9 = tfa.layers.GroupNormalization(groups= 8, axis = 3)(c9)

    pre_output= tf.keras.layers.Conv2D(n_classes,(1,1),activation='softmax')(c9)
    outputs = tf.keras.layers.Reshape([(IMG_HEIGHT*IMG_WIDTH),n_classes])(pre_output) 

    model = tf.keras.Model(inputs=[inputs], outputs =[outputs])
    
    return model



