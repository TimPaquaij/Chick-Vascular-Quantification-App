from telnetlib import OUTMRK
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from patchify import patchify, unpatchify
from Instance_semantic_segmentation_model import unet_multiclass
from mpl_toolkits.axes_grid1 import make_axes_locatable
matplotlib.use('TkAgg')


def fig2data(fig):
        """
        @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
        @param fig a matplotlib figure
        @return a numpy 3D array of RGBA values
        """
        # draw the renderer
        fig.canvas.draw ( )

        # Get the RGBA buffer from the figure
        w,h = fig.canvas.get_width_height()
        print(w,h)
        buf = np.fromstring (fig.canvas.tostring_argb(), dtype=np.uint8 )
        buf.shape = ( w, h,4 )

        # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
        buf = np.roll ( buf, 3, axis = 2 )
        buf = cv2.cvtColor(buf,cv2.COLOR_RGB2BGR)
        return buf 


model =unet_multiclass(19, 512,512,3) #loading model
model.load_weights('multiclass_model_512_test.h5') # adding wheights
video = cv2.VideoCapture(0)
diameter_width_array = [0,2,5,7,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110]
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
close = 'no'
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use the lower case
out = cv2.VideoWriter('output_def.mp4', fourcc, 20,(800,800))
pixel_width =1.43
digital_zoom = 10
while True:
    _, frame = video.read()
    if frame is None:
        break
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    images = patchify(frame,(512,512,3), step=512)
    print(images.shape)
    input_images = images.reshape(images.shape[0]*images.shape[1], 512,512,3)
    print(input_images.shape)
    prediction = (model.predict(input_images)) #Making prediction
    prediction = prediction.reshape(input_images.shape[0],512,512,19)
    print(prediction.shape)
#
 #r shaping output image
    predicted_img=np.argmax(prediction, axis=3)
    new_predicted_img= np.ones(predicted_img.shape,dtype = float)
    for k in range(predicted_img.shape[0]):    
        for i in range(predicted_img.shape[1]):
            for j in range(predicted_img.shape[2]):
                class_nr = predicted_img[k,i,j]
                new_predicted_img[k,i,j] = diameter_width_array[class_nr]q
#
    
    new_predicted_img = np.array(new_predicted_img)*pixel_width*digital_zoom
    new_predicted_img =np.round(new_predicted_img,1)
    reshape_prediction = new_predicted_img.reshape((images.shape[0],images.shape[1],512,512)) #reshaping for unpatchify
    segmented_frame = unpatchify(reshape_prediction,(512*images.shape[0],512*images.shape[1])) #unpatchify for total image
    fig_input,ax1 = plt.subplots(nrows =2, ncols = 1,figsize = (8,8))
    i = ax1[0].imshow(segmented_frame)
    ax1[0].get_xaxis().set_visible(False)     
    divider = make_axes_locatable(ax1[0])
    cax = divider.new_vertical(size = '5%', pad = 0.01, pack_start = True)
    fig_input.add_axes(cax)
    cbar =plt.colorbar(i,orientation='horizontal', cax= cax) 
    cbar.ax.set_xlabel('Diameter (\u03BCm)')
#
    i = ax1[1].imshow(frame)
    ax1[1].get_xaxis().set_visible(False)
    ax1[1].get_yaxis().set_visible(False)         
    img = fig2data(fig_input)
    cv2.imshow('Camera',img)
    plt.close()
    out.write(img)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
out.release()
cv2.destroyAllWindows()
