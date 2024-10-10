import numpy as np
from core.CV.diameter_shortest_distance import Thresh, Mid_line

def split_end_point(img):
    #Detect the diffrent points which form the middle_line
    end_point = []
    lonely_point = []
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j] == 255:
                branch = np.where(img[i-1:i+2:,j-1:j+2]==255) #Detects all the surrounding pixels classified as mid-line
                branch = np.array(branch)
                if branch.size <3: # if pixels is only surounded by background the position is saved
                    a= i,j
                    lonely_point.append(a)
                elif branch.size > 6: #If the pixels is surounded by more then two pixels it will be classified als mid-point
                    img[i][j] = 5
                    

            
    for a in lonely_point: #All the saved pixel position which where lonley will be set to background
        i,j = a
        img[i][j] = 0

    return img

def branch(img,threshold_skl, clipLimit,tile,sigma,YLength,m, thresh):
    img_thresh = Thresh(img,clipLimit,tile,sigma,YLength,m,thresh) #uses the same threshold as in diameter_shortest_distance
    mid_line = Mid_line(img_thresh, threshold_skl) #Same mid-line detection
    changed_mid_line = split_end_point(mid_line) #Function determine the split-point in the mid-line structure

    
    return changed_mid_line

#image = cv2.imread('training/frame 4/Total_img.png')
#print(image)
#image_patches = patchify(image,(128,128,3), step = 128)
#for i in range(image_patches.shape[0]):
#    for j in range(image_patches.shape[1]):
#        image_patch = image_patches[i,j,0,:,:,:]
#        img_branch=branch(image_patch,2,6,1)
#        plt.imshow(img_branch)  
#        plt.show()
#
    
