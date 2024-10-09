from patchify import patchify, unpatchify
import numpy as np
from diameter_shortest_distance import diameter_short
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
import imageio as io
from tkinter.filedialog import askopenfile, askopenfilename





        #Reading the image and plotting in the input graph
image= cv2.imread('Images Hayear/H-19.TIF')
image_patches = patchify(image,(512,512,3),step=256)    
#print(image_patches.shape)
#image = image_patches[2,5,0,:,:,:]
#image_diameter= diameter_short(image,40,-0.0051,1,1,8,31,180,250)  
#plt.subplot(2,1,1)
#plt.imshow(image)
#plt.subplot(2,1,2)
#plt.imshow(image_diameter)
#plt.show()
#f=1
#for a in [2,4]:
#        for b in [2,4]:
#                ax = fig.add_subplot(2,4,f)
#                image = image_patches[a,b,0,:,:,:] 
#
#                ax.imshow(image)
#                ax.title.set_text(str(a)+str(b)) 
#                f+=1
#                image_diameter= diameter_short(image, 30,-0.0051,1,1,6,40,180,250) 
#                ax = fig.add_subplot(2,4,f)
#                ax.imshow(image_diameter)
#                ax.title.set_text(str(a)+str(b))
#                print('done')
#                f+=1
#
#plt.show()     
for l in tqdm(range(image_patches.shape[0])):
        for k in tqdm(range(image_patches.shape[1])):
                image = image_patches[l,k,0,:,:,:]
                image_diameter= diameter_short(image,30,-0.0051,1,1,8,31,180,250)   #img,radius,threshold_skl,clipLimit,tile,sigma, YLength,m,e,threshold_thresh
                cv2.imwrite('training_512/19_color/'+str(l)+'_'+str(k)+'_19_normal.png',image)
                cv2.imwrite('training_512/19_color/'+str(l)+'_'+str(k)+'_19_diameter.png',image_diameter)
                print('done')












 