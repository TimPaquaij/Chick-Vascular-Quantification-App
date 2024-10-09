
from collections import Counter
from pandas import DataFrame
import numpy as np
import cv2
from skimage.transform import resize
filepath = 'Images 2/1.TIF'
        

    #Reading the image and plotting in the input graph
image= cv2.imread(filepath)
image = np.array(image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
size_img = image.shape
new_height = int(size_img[0]/512)*512
new_width = int(size_img[1]/512)*512
new_image = resize(image,(new_height,new_width,3),mode='constant',preserve_range='true') #resize image into patchable size
new_image = np.array(new_image,  dtype = 'uint8')


list_1 = new_image.flatten().tolist()


counts = Counter(list_1)
Numbers = [*counts.keys()]
Values = [*counts.values()]

number = np.array(Numbers)
values = np.array(Values)

print(number.shape)
print(values.shape)

