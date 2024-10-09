
from PIL import Image, ImageStat
import numpy as np
import cv2
from skimage import morphology as mp
import shapely
import scipy
import skimage
from shapely.geometry import LineString
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage import gaussian_laplace
from points_from_line_extraction import get_line
from Thresholding import process, grayStretch,adjust_gamma,build_filters2
import statistics
from skimage.filters.rank import core_cy_3d

def Thresh(img,clipLimit, tile,sigma, YLength,m,threshold_thresh):
    
    
    # Green channel extraction
    grayImg = cv2.split(img)[1]
    # Gaussian filtering
    blurImg = cv2.GaussianBlur(grayImg, (5, 5), 0)
    # 5x5 kernel
    # CLAHE Light equalization + contrast enhancement
    clahe = cv2.createCLAHE(clipLimit, tileGridSize=(tile,tile))
    claheImg = clahe.apply(blurImg)

    #Gama correction
    preMFImg = adjust_gamma(claheImg, gamma=1.5)

    filters = build_filters2(sigma, YLength)
    # showKern(filters) for Gaussian Matched Filtering
    gaussMFImg = process(preMFImg, filters)
    # Gray Scretching for linearalisation
    grayStretchImg = grayStretch(gaussMFImg, m=m / 255)
    # Binarization
    ret1, th1 = cv2.threshold(grayStretchImg, threshold_thresh, 255, cv2.THRESH_OTSU)
    predictImg = th1.copy() # Segment the image into a binary image, where vessel has a value op  1 and backround is 0
    return claheImg, gaussMFImg, predictImg


def Edge(img): #Use the module canny for edge detection
    img = np.array(img, dtype=np.uint8) #convert to 8 bit array
    min_threshold = img.max() *0.5 #Minimal THreshoolding
    max_threshold= img.max()    #Maximum hresholding
    edges = cv2.Canny(img,min_threshold,max_threshold,L2gradient= True) #Canny edge detectioon
    return edges 

def Mid_line(img,threshold_skl): #Use a distance transfom and a gausian filter to detect the midle line.
    img_new = np.array(img, dtype = np.uint8)  #convert to 8 bit array
    distance_t = distance_transform_edt(img_new,100) #Transfor to a ecludean distance transform
    norm_distance = distance_t/distance_t.max() #Determmine the normalisated distance transform
    middle = gaussian_laplace(norm_distance,2) # Use a gaussian laplace fuction for stretching
    ret4, img_thresh2 = cv2.threshold(middle,threshold_skl,1,cv2.THRESH_BINARY_INV) #Inverse binarisation
    mid_line = mp.skeletonize(img_thresh2, method = 'lee') #Skeletonize for optimal mid-line detection

    
    return mid_line


def average_area(img): #Determine the average area for image smoothing
    distances = np.unique(img) #detects every unique number in image
    new_img = np.zeros((img.shape)) #makes an emty array 
    for k in distances: #Goes for every unque number throught whole image
        for i in range(img.shape[0]): 
            for j in range(img.shape[1]):
                if img[i,j] ==k: #search where this number is classifieed
                    branch = np.where(img[i-1:i+2:,j-1:j+2]==k) #checks how often it apprears in the naberhood
                    branch = np.array(branch) #transfomrs output to array
                    if 0< branch.size < 12: #if the number appears les then 3 ties in the naberhood it checks which number appears often en changes it to that number
                        number = ImageStat.Stat(img[i-1:i+2:,j-1:j+2].tolist()).count
                        number=statistics.mode(number[0])
                        new_img[i,j] = number
                    else: #It will  hold his original number
                        new_img[i,j] = k
    
    return new_img

                
                


def points_mid_line(img): #Makes  a list for all pixels wich where classified as mid_line
    point = []
    for i in range(0,img.shape[0],1):
        for j in range(0,img.shape[1],1):
            if img[i,j] == 255:
                a = i,j
                point.append(a)
            else:
                False
    
    
    return point

def diameter_short(img,radius,threshold_skl,clipLimit,tile,sigma, YLength,m,threshold_thresh): #Radius, threshold_skl, contrast cliplimit, tile size, sigma, filtersize,mean, thresholdthresh
    
    
    img_height =  512
    img_width  =  512
    
    claheImg, gaussMFImg,img_thresh= Thresh(img,clipLimit,tile,sigma,YLength,m,threshold_thresh) # Function for threesholdiing
    img_thresh_def = img_thresh.copy()
    edges  = Edge(img_thresh)
    
    
    mid_line = Mid_line(img_thresh, threshold_skl) # Fuction to make mid-line
    
    point = points_mid_line(mid_line) #Function to covert mid-line ot a list of points
   
    
    #fig = plt.figure()
    #ax = fig.add_subplot(1,1,1)
    #plt.imshow(mid_line_edge, aspect= 'auto')
    for z in point:   #Goes throught every point of the middle line
             
        i,j = z
        line_length = radius
        distance_list= []
        points_list=[]
        for thetha in range(0,180,15): #Makes multiple lines wich go throught the middle point with diffrent angles
            Both_points = []
            i_right =i + (line_length*np.sin((2*np.pi/360)*thetha)) 
            j_right =j + (line_length*np.cos((2*np.pi/360)*thetha))
            i_left =i - (line_length*np.sin((2*np.pi/360)*thetha))
            j_left =j - (line_length*np.cos((2*np.pi/360)*thetha))
            new_point_right = [int(i_right),int(j_right)]
            new_point_left = [int(i_left),int(j_left)]
            #ax.plot([new_point_left[1],new_point_right[1]],[new_point_left[0],new_point_right[0]])
            line_points = get_line(new_point_left[1],new_point_left[0],new_point_right[1],new_point_right[0]) #Find everry point where the line goes throught 
            line_points = np.array(line_points)
            for cor in line_points: #Detect where the line crosses a point of the edge
                a,b = cor 
                if 0<=a<=(img_width-1):
                    if 0<=b<(img_height-1):
                        if edges[b,a] ==255:
                            Both_points.append([a,b])
                        else:
                            False

            if len(Both_points) == 2:
                 #If the line crosses two sides of the edge it calculates the distance
                p5,p6 = Both_points
                p5[0], p5[1], p6[0],p6[1] = int(p5[0]), int(p5[1]), int(p6[0]),int(p6[1])
                line_3 = LineString([(p5[0],p5[1]),(p6[0],p6[1])])
                distance = line_3.length
                if distance < 3:
                    False #For very low values of the distance the line went throught two pixels of the same side and have to be deleted  
                else: #Save the distance in a list, and the reference points in another list at the same position
                    distance_list.append(distance)
                    final_points = get_line(p5[0], p5[1], p6[0],p6[1])
                    points_list.append(final_points)
    
    
        if not distance_list: # Checks if there are multiple distances which could be detected
            False
        else: # Calculated the position of the smallest distance
            distance_array = np.array(distance_list)
            distance_point= np.where(distance_list==min(distance_array))
            if len(distance_point[0]) >= 2: #if there are multple exacly the same distances it choses the first one
                distance_point = int(distance_point[0][0])
            else:
                distance_point = int(distance_point[0])    
                                                                  #Makes an interger of the position and search for the reference point of that line
            points_array =  np.array(points_list,dtype= object)
            points_def  = points_array[distance_point]
            if min(distance_array) <2: #Rounds it for smoothing the image
                distance = 0
            elif min(distance_array) <4:
                distance = 2
            elif min(distance_array)<7:
                distance = 5
            elif min(distance_array)<9:
                distance = 7
            elif min(distance_array)<12:
                distance = 10
            elif min(distance_array)<17:
                distance = 15
            elif min(distance_array)<22:
                distance = 20
            elif min(distance_array)<27:
                distance = 25
            elif min(distance_array)<32:
                distance = 30
            elif min(distance_array)<37:
                distance = 35
            elif min(distance_array)<42:
                distance = 40
            elif min(distance_array)<47:
                distance = 45
            elif min(distance_array)<52:
                distance = 50
            elif min(distance_array)<57:
                distance = 55
            elif min(distance_array)<62:
                distance = 60
            elif min(distance_array)<67:
                distance = 65    
            elif min(distance_array)<72:
                 distance = 70   
            elif min(distance_array)<77:
                 distance = 75    
            elif min(distance_array)<82:
                distance = 80
            elif min(distance_array)<87:
                distance = 85    
            elif min(distance_array)<92:
                distance = 90    
            elif min(distance_array)<97:
                distance = 95
            elif min(distance_array)<102:
                distance = 100    
            elif min(distance_array)<107:
                distance = 105          
            else:
                distance = 110

            
        
            #heel = int(deel)
            #rest = deel-heel
            #if rest >= 0.5:
            #    afronding = (1 + heel)*5
#
            #else:
            #    afronding = heel*5
            #
            #ax[2].plot((p7[0],p8[0]),(p7[1],p8[1]))
            for point_def in points_def:
                i,j = point_def
                img_thresh[j-1:j+2,i-1:i+2] = np.where(img_thresh[j-1:j+2,i-1:i+2]== 255,distance,img_thresh[j-1:j+2,i-1:i+2]) #Colors every pixel refering to that line with the distance value.
    
    for i in range(img_thresh.shape[0]):
        for j in range(img_thresh.shape[0]):
            img_thresh[i,j] = np.where(img_thresh[i,j]== 255,0,img_thresh[i,j]) #Pixels which hassn't been classiefied with a distacnce will be set to background

    img_def = average_area(img_thresh) #For further smoothing a average check is done to see if surounding pixels have the same value
    #plt.savefig('Images_def/Short_distance.png', dpi =400)

    #img_def =average_area(img_thresh)    




    



    
    return img_def