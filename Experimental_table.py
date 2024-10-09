from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2
from Instance_semantic_segmentation_model import unet_multiclass
from pandas import DataFrame
import seaborn as sns
from collections import Counter
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import pandas as pd
import io
pixel_size =1.43
digital_zoom = 1


plt.rcParams.update({'font.size': 8})

img_1 = cv2.cvtColor(cv2.imread('Figure 2/1_4_11_normal.png'),cv2.COLOR_BGR2RGB)
img_1_diameter =cv2.imread('Figure 2/1_4_11_diameter.png',1)
def extract_data(img_diameter):
    part_list = []
    sumation =0
    image_list = img_diameter.flatten().tolist()
    number_dict = Counter(image_list).keys()
    amount_dict = Counter(image_list).values()
    number_list = [*number_dict]
    amount_list = [*amount_dict]
    diameter_width_array =np.array([2,5,7,10,15,20,25,30,35,40,45,50,55,60])*pixel_size*digital_zoom
    diameter_width_array =np.round(diameter_width_array,1)
    for a in diameter_width_array:
        if a not in number_list:
            number_list.append(a)
            amount_list.append(0)
    #array_number_list = np.unique(image) #Creating list with unique values
    #for i in range(1,len(array_number_list),1):
    #    number_list.append(array_number_list[i]) #Selecting diameter without classes 0 and 255
    #    amount = image_list.count(array_number_list[i]) #Counts how much of every unique values is inside the input image not 0
    #    amount_list.append(amount)  #Saves these value in list
    total = sum(amount_list[1:]) #Counts total amount of pixels classifies as vessel
    #
    for i in amount_list[1:]: #For every value in list counts the precentage of total vascular area
        part = i/total
        part_list.append(part)
    for a in range(len(amount_list)-1):
        amount= number_list[a+1]*amount_list[a+1]
        sumation = sumation + amount
    average = sumation/total
    data_diameter = {'Diameter (\u03BCm)': np.array(number_list[1:]), 'Amount of Pixels' : np.array(amount_list[1:]), 'Part of total vessel area' : np.array(part_list), 'Average value' :average} #Making dictionary
    df1 = DataFrame(data_diameter,columns=['Diameter (\u03BCm)','Amount of Pixels','Part of total vessel area', 'Average value']) #Making dataframe of dictionary
    return df1

img_1 = np.array(img_1)
img_1_diameter = np.array(img_1_diameter)*pixel_size*digital_zoom
img_1_diameter = np.round(img_1_diameter,1)




img_2 = cv2.cvtColor(cv2.imread('Figure 2/1_0_1_normal.png'),cv2.COLOR_BGR2RGB)
img_2_diameter =cv2.imread('Figure 2/1_0_1_diameter.png',1)
img_2 = np.array(img_2)
img_2_diameter = np.round(np.array(img_2_diameter)*pixel_size*digital_zoom,1)

img_3 = cv2.cvtColor(cv2.imread('Figure 2/0_4_19_normal.png'),cv2.COLOR_BGR2RGB)
img_3_diameter =cv2.imread('Figure 2/0_4_19_diameter.png',1)
img_3 = np.array(img_3)
img_3_diameter = np.round(np.array(img_3_diameter)*pixel_size*digital_zoom,1)


img_4 = cv2.cvtColor(cv2.imread('Figure 2/1_1_8_normal.png'),cv2.COLOR_BGR2RGB)
img_4_diameter =cv2.imread('Figure 2/1_1_8_diameter.png',1)
img_4 = np.array(img_4)
img_4_diameter = np.round(np.array(img_4_diameter)*pixel_size*digital_zoom,1)




img_5 = cv2.cvtColor(cv2.imread('Figure 2/2_2_19_normal.png'),cv2.COLOR_BGR2RGB)
img_5_diameter =cv2.imread('Figure 2/2_2_19_diameter.png',1)
img_5 = np.array(img_5)
img_5_diameter = np.round(np.array(img_5_diameter)*pixel_size*digital_zoom,1)

img_6 = cv2.cvtColor(cv2.imread('Figure 2/2_1_8_normal.png'),cv2.COLOR_BGR2RGB)
img_6_diameter =cv2.imread('Figure 2/2_1_8_diameter.png',1)
img_6 = np.array(img_6)
img_6_diameter = np.round(np.array(img_6_diameter)*pixel_size*digital_zoom,1)
#
#img_7 = cv2.imread('training_512/12_bmp/5_5_12_normal.png')
#img_7_diameter =cv2.imread('training_512/12_bmp/5_5_12_diameter.png')
#img_7 = np.array(img_7)
#img_7_diameter = np.array(img_7_diameter)
#
#img_8 = cv2.imread('training_512/12_bmp/2_3_12_normal.png')
#img_8_diameter =cv2.imread('training_512/12_bmp/2_3_12_diameter.png')
#img_8 = np.array(img_8)
#img_8_diameter = np.array(img_8_diameter)



prediction_array = np.ones((6,512,512,3), dtype = int)
prediction_array[0,:,:,:] = img_1
prediction_array[1,:,:,:] = img_2
prediction_array[2,:,:,:] = img_3
prediction_array[3,:,:,:] = img_4
prediction_array[4,:,:,:] = img_5
prediction_array[5,:,:,:] = img_6

diameter_width_array = [0,2,5,7,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110]

model =unet_multiclass(19, 512,512,3) #loading model
model.load_weights('multiclass_model_'+str(512)+'_test.h5') # adding wheights
prediction = (model.predict(prediction_array)) #Making prediction
prediction = prediction.reshape(prediction_array.shape[0],512,512,19)

 #reshaping output image
predicted_img=np.argmax(prediction, axis=3)
new_predicted_img= np.ones(predicted_img.shape,dtype = float)
print(new_predicted_img.shape)
for k in range(predicted_img.shape[0]):    
    for i in range(predicted_img.shape[1]):
        for j in range(predicted_img.shape[2]):
            class_nr = predicted_img[k,i,j]
            new_predicted_img[k,i,j] = diameter_width_array[class_nr]
#detemning output image
new_predicted_img = np.array(new_predicted_img)*digital_zoom*pixel_size
new_predicted_img =np.round(new_predicted_img,1)

fig_number = 0
max_value = 60*digital_zoom*pixel_size





writer = pd.ExcelWriter('Excel_Chart_data.xlsx', engine='xlsxwriter') 


fig, ax = plt.subplots()
ax.imshow(img_1,aspect='auto')
ax.set_title('Raw Image Smal Vessels',size =10)
fig.savefig('Table 2 Images/Raw_Image_1.png',dpi=300)
plt.close()

fig, ax = plt.subplots()
ax.get_xaxis().set_visible(False)
i= ax.imshow(img_1_diameter[:,:,0],interpolation='nearest',aspect='auto')
ax.set_title('Esimated diameter (Algorithm)')
divider = make_axes_locatable(ax)
cax = divider.new_vertical(size = '5%', pad = 0.01, pack_start = True)
fig.add_axes(cax)
cbar =plt.colorbar(i,orientation='horizontal', cax= cax) 
cbar.ax.set_xlabel('\u03BCm')
i.set_clim(0,max_value)
fig.savefig('Table 2 Images/Estimated_Image_1.png',dpi=300)
plt.close()




fig, ax = plt.subplots()
ax.get_xaxis().set_visible(False)
i =ax.imshow(new_predicted_img[0], aspect = 'auto',interpolation='nearest')
ax.set_title('Predicted Diameter (Unet)')
divider = make_axes_locatable(ax)
cax = divider.new_vertical(size = '5%', pad = 0.01, pack_start = True)
fig.add_axes(cax,size = 8)
cbar =plt.colorbar(i,orientation='horizontal', cax= cax)
cbar.ax.set_xlabel('\u03BCm',size =8)
i.set_clim(0,max_value)
fig.savefig('Table 2 Images/Predicted_Image_1.png',dpi=300)
plt.close()


fig, ax = plt.subplots()
ax.set_title('Diameter Estimation')
dataframe = extract_data(img_1_diameter[:,:,0])
print({'Average estimation 1'}, dataframe['Average value'][0])
color = sns.color_palette("viridis",14)
ax = sns.barplot(x='Diameter (\u03BCm)', y='Part of total vessel area', data=dataframe,ax=ax,palette=color[1:],ci=None)
dataframe.to_excel(writer, sheet_name = "Estimation 1",index =True, header=True)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right",size = 8)
fig.savefig('Table 2 Images/Chart_Estimation_Image_1.png',dpi=300)
plt.close()

fig, ax = plt.subplots()
ax.set_title('Diameter Prediction')
dataframe = extract_data(new_predicted_img[0])
print({'Average prediction 1'}, dataframe['Average value'][0])
ax = sns.barplot(x='Diameter (\u03BCm)', y='Part of total vessel area', data=dataframe,ax=ax,palette=color[1:],ci=None)
dataframe.to_excel(writer, sheet_name = "Prediction 1",index =True, header=True)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right",size = 8)
fig.savefig('Table 2 Images/Chart_Prediction_Image_1.png',dpi=300)
plt.close()



fig, ax = plt.subplots() 
ax.imshow(img_2,aspect='auto')
ax.set_title('Raw Image Large Vessels',size =10)
fig.savefig('Table 2 Images/Raw_Image_2.png',dpi=300)
plt.close()


fig, ax = plt.subplots()
ax.get_xaxis().set_visible(False)
i= ax.imshow(img_2_diameter[:,:,0],interpolation='nearest',aspect='auto')
ax.set_title('Esimated diameter (Algorithm)')
divider = make_axes_locatable(ax)
cax = divider.new_vertical(size = '5%', pad = 0.01, pack_start = True)
fig.add_axes(cax)
cbar =plt.colorbar(i,orientation='horizontal', cax= cax) 
cbar.ax.set_xlabel('\u03BCm')
i.set_clim(0,max_value)
fig.savefig('Table 2 Images/Estimated_Image_2.png',dpi=300)
plt.close()


fig, ax = plt.subplots()
ax.get_xaxis().set_visible(False)
i =ax.imshow(new_predicted_img[1], aspect = 'auto',interpolation='nearest')
ax.set_title('Predicted Diameter (Unet)')
divider = make_axes_locatable(ax)
cax = divider.new_vertical(size = '5%', pad = 0.01, pack_start = True)
fig.add_axes(cax,size = 8)
cbar =plt.colorbar(i,orientation='horizontal', cax= cax) 
cbar.ax.set_xlabel('\u03BCm',size =8)
i.set_clim(0,max_value)
fig.savefig('Table 2 Images/Predicted_Image_2.png',dpi=300)
plt.close()


fig, ax = plt.subplots()
ax.set_title('Diameter Estimation')
dataframe = extract_data(img_2_diameter[:,:,0])
print({'Average estimation 2'}, dataframe['Average value'][0])
dataframe.to_excel(writer, sheet_name = "Estimation 2",index =True, header=True)
ax = sns.barplot(x='Diameter (\u03BCm)', y='Part of total vessel area', data=dataframe,ax=ax,palette=color[1:],ci=None)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right",size = 8)
fig.savefig('Table 2 Images/Chart_Estimation_Image_2.png',dpi=300)
plt.close()





fig, ax = plt.subplots()
ax.set_title('Diameter Prediction')
dataframe = extract_data(new_predicted_img[1])
print({'Average prediction 2'}, dataframe['Average value'][0])
dataframe.to_excel(writer, sheet_name = "Prediction 2",index =True, header=True)
ax = sns.barplot(x='Diameter (\u03BCm)', y='Part of total vessel area', data=dataframe,ax=ax,palette=color[1:],ci=None)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right",size = 8)
fig.savefig('Table 2 Images/Chart_Prediction_Image_2.png',dpi=300)
plt.close()



fig, ax = plt.subplots()   
ax.imshow(img_3,aspect='auto')
ax.set_title('Raw Image Mixed Vessels',size =10)
fig.savefig('Table 2 Images/Raw_Image_3.png',dpi=300)
plt.close()


fig, ax = plt.subplots()
ax.get_xaxis().set_visible(False)
i= ax.imshow(img_3_diameter[:,:,0],interpolation='nearest',aspect='auto')
ax.set_title('Esimated diameter (Algorithm)')
divider = make_axes_locatable(ax)
cax = divider.new_vertical(size = '5%', pad = 0.01, pack_start = True)
fig.add_axes(cax)
cbar =plt.colorbar(i,orientation='horizontal', cax= cax) 
cbar.ax.set_xlabel('\u03BCm')
i.set_clim(0,max_value)
fig.savefig('Table 2 Images/Estimated_Image_3.png',dpi=300)
plt.close()


fig, ax = plt.subplots()
ax.get_xaxis().set_visible(False)
i =ax.imshow(new_predicted_img[2], aspect = 'auto',interpolation='nearest')
ax.set_title('Predicted Diameter (Unet)')
divider = make_axes_locatable(ax)
cax = divider.new_vertical(size = '5%', pad = 0.01, pack_start = True)
fig.add_axes(cax,size = 8)
cbar =plt.colorbar(i,orientation='horizontal', cax= cax) 
cbar.ax.set_xlabel('\u03BCm',size =8)
i.set_clim(0,max_value)
fig.savefig('Table 2 Images/Predicted_Image_3.png',dpi=300)
plt.close()


fig, ax = plt.subplots()
ax.set_title('Diameter Estimation')
dataframe = extract_data(img_3_diameter[:,:,0])
print({'Average estimation 3'}, dataframe['Average value'][0])
dataframe.to_excel(writer, sheet_name = "Estimation 3",index =True, header=True)
ax = sns.barplot(x='Diameter (\u03BCm)', y='Part of total vessel area', data=dataframe,ax=ax,palette=color[1:],ci=None)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right",size = 8)
fig.savefig('Table 2 Images/Chart_Estimation_Image_3.png',dpi=300)
plt.close()




fig, ax = plt.subplots()
ax.set_title('Diameter Prediction')
dataframe = extract_data(new_predicted_img[2])
print({'Average prediction 3'}, dataframe['Average value'][0])
dataframe.to_excel(writer, sheet_name = "Prediction 3",index =True, header=True)
ax = sns.barplot(x='Diameter (\u03BCm)', y='Part of total vessel area', data=dataframe,ax=ax,palette=color[1:],ci=None)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right",size = 8)
fig.savefig('Table 2 Images/Chart_Prediction_Image_3.png',dpi=300)
plt.close()



fig, ax = plt.subplots() 
ax.imshow(img_4,aspect='auto')
ax.set_title('Raw Image Bifurcated Vessels',size =10)
fig.savefig('Table 2 Images/Raw_Image_4.png',dpi=300)
plt.close()



fig, ax = plt.subplots()
ax.get_xaxis().set_visible(False)
i= ax.imshow(img_4_diameter[:,:,0],interpolation='nearest',aspect='auto')
ax.set_title('Esimated diameter (Algorithm)')
divider = make_axes_locatable(ax)
cax = divider.new_vertical(size = '5%', pad = 0.01, pack_start = True)
fig.add_axes(cax)
cbar =plt.colorbar(i,orientation='horizontal', cax= cax) 
cbar.ax.set_xlabel('\u03BCm')
i.set_clim(0,max_value)
fig.savefig('Table 2 Images/Estimated_Image_4.png',dpi=300)
plt.close()


fig, ax = plt.subplots()
ax.get_xaxis().set_visible(False)
i =ax.imshow(new_predicted_img[3], aspect = 'auto',interpolation='nearest')
ax.set_title('Predicted Diameter (Unet)')
divider = make_axes_locatable(ax)
cax = divider.new_vertical(size = '5%', pad = 0.01, pack_start = True)
fig.add_axes(cax,size = 8)
cbar =plt.colorbar(i,orientation='horizontal', cax= cax) 
cbar.ax.set_xlabel('\u03BCm',size =8)
i.set_clim(0,max_value)
fig.savefig('Table 2 Images/Predicted_Image_4.png',dpi=300)
plt.close()

fig, ax = plt.subplots()
ax.set_title('Diameter Estimation')
dataframe = extract_data(img_4_diameter[:,:,0])
print({'Average estimation 4'}, dataframe['Average value'][0])
dataframe.to_excel(writer, sheet_name = "Estimation 4",index =True, header=True)
ax = sns.barplot(x='Diameter (\u03BCm)', y='Part of total vessel area', data=dataframe,ax=ax,palette=color[1:],ci=None)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right",size = 8)
fig.savefig('Table 2 Images/Chart_Estimation_Image_4.png',dpi=300)
plt.close()



fig, ax = plt.subplots()
ax.set_title('Diameter Prediction')
dataframe = extract_data(new_predicted_img[3])
print({'Average prediction 4'}, dataframe['Average value'][0])
dataframe.to_excel(writer, sheet_name = "Prediction 4",index =True, header=True)
ax = sns.barplot(x='Diameter (\u03BCm)', y='Part of total vessel area', data=dataframe,ax=ax,palette=color[1:],ci=None)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right",size = 8)
fig.savefig('Table 2 Images/Chart_Prediction_Image_4.png',dpi=300)
plt.close()




fig, ax = plt.subplots()
ax.imshow(img_5,aspect='auto')
ax.set_title('Raw Image Tortuous Vessels',size =10)
fig.savefig('Table 2 Images/Raw_Image_5.png',dpi=300)
plt.close()



fig, ax = plt.subplots()
ax.get_xaxis().set_visible(False)
i= ax.imshow(img_5_diameter[:,:,0],interpolation='nearest',aspect='auto')
ax.set_title('Esimated diameter (Algorithm)')
divider = make_axes_locatable(ax)
cax = divider.new_vertical(size = '5%', pad = 0.01, pack_start = True)
fig.add_axes(cax)
cbar =plt.colorbar(i,orientation='horizontal', cax= cax) 
cbar.ax.set_xlabel('\u03BCm')
i.set_clim(0,max_value)
fig.savefig('Table 2 Images/Estimated_Image_5.png',dpi=300)
plt.close()



fig, ax = plt.subplots()
ax.get_xaxis().set_visible(False)
i =ax.imshow(new_predicted_img[4], aspect = 'auto',interpolation='nearest')
ax.set_title('Predicted Diameter (Unet)')
divider = make_axes_locatable(ax)
cax = divider.new_vertical(size = '5%', pad = 0.01, pack_start = True)
fig.add_axes(cax,size = 8)
cbar =plt.colorbar(i,orientation='horizontal', cax= cax) 
cbar.ax.set_xlabel('\u03BCm',size =8)
i.set_clim(0,max_value)
fig.savefig('Table 2 Images/Predicted_Image_5.png',dpi=300)
plt.close()


fig, ax = plt.subplots()
ax.set_title('Diameter Estimation')
dataframe = extract_data(img_5_diameter[:,:,0])
print({'Average estimation 5'}, dataframe['Average value'][0])
dataframe.to_excel(writer, sheet_name = "Estimation 5",index =True, header=True)
ax = sns.barplot(x='Diameter (\u03BCm)', y='Part of total vessel area', data=dataframe,ax=ax,palette=color[1:],ci=None)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right",size = 8)
fig.savefig('Table 2 Images/Chart_Estimation_Image_5.png',dpi=300)
plt.close()



fig, ax = plt.subplots()
ax.set_title('Diameter Prediction')
dataframe = extract_data(new_predicted_img[4])
print({'Average prediction 5'}, dataframe['Average value'][0])
dataframe.to_excel(writer, sheet_name = "Prediction 5",index =True, header=True)
ax = sns.barplot(x='Diameter (\u03BCm)', y='Part of total vessel area', data=dataframe,ax=ax,palette=color[1:],ci=None)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right",size = 8)
fig.savefig('Table 2 Images/Chart_Prediction_Image_5.png',dpi=300)
plt.close()
writer.save()


#ax = fig.add_subplot(5, 3, 4)   
#ax.imshow(img_2,aspect='auto')
#ax.set_title('Raw Image Large Vessels',size =10)
#
#ax = fig.add_subplot(5, 3, 5)
#ax.get_xaxis().set_visible(False)
#i= ax.imshow(img_2_diameter[:,:,0],interpolation='nearest',aspect='auto')
#ax.set_title('Esimated diameter (Algorithm)')
#divider = make_axes_locatable(ax)
#cax = divider.new_vertical(size = '5%', pad = 0.01, pack_start = True)
#fig.add_axes(cax)
#cbar =plt.colorbar(i,orientation='horizontal', cax= cax) 
#cbar.ax.set_xlabel('\u03BCm')
#
#ax = fig.add_subplot(5, 3,6)
#ax.get_xaxis().set_visible(False)
#i =ax.imshow(new_predicted_img[1]*pixel_size*digital_zoom, aspect = 'auto',interpolation='nearest')
#ax.set_title('Predicted Diameter (Unet)')
#divider = make_axes_locatable(ax)
#cax = divider.new_vertical(size = '5%', pad = 0.01, pack_start = True)
#fig.add_axes(cax)
#cbar =plt.colorbar(i,orientation='horizontal', cax= cax) 
#cbar.ax.set_xlabel('\u03BCm')
#
#ax = fig.add_subplot(5, 3, 7)   
#ax.imshow(img_3,aspect='auto')
#ax.set_title('Raw Image Mixed Vessels',size =10)
#
#ax = fig.add_subplot(5, 3, 8)
#ax.get_xaxis().set_visible(False)
#i= ax.imshow(img_3_diameter[:,:,0],interpolation='nearest', aspect='auto')
#ax.set_title('Esimated diameter (Algorithm)')
#divider = make_axes_locatable(ax)
#cax = divider.new_vertical(size = '5%', pad = 0.01, pack_start = True)
#fig.add_axes(cax)
#cbar =plt.colorbar(i,orientation='horizontal', cax= cax) 
#cbar.ax.set_xlabel('\u03BCm')
#
#ax = fig.add_subplot(5, 3,9)
#ax.get_xaxis().set_visible(False)
#i =ax.imshow(new_predicted_img[2]*pixel_size*digital_zoom, aspect = 'auto',interpolation='nearest')
#ax.set_title('Predicted Diameter (Unet)')
#divider = make_axes_locatable(ax)
#cax = divider.new_vertical(size = '5%', pad = 0.01, pack_start = True)
#fig.add_axes(cax)
#cbar =plt.colorbar(i,orientation='horizontal', cax= cax) 
#cbar.ax.set_xlabel('\u03BCm')
#
#
#ax = fig.add_subplot(5, 3, 10)   
#ax.imshow(img_4,aspect='auto')
#ax.set_title('Raw Image Bifurcated Vessels',size =10)
#
#ax = fig.add_subplot(5, 3, 11)
#ax.get_xaxis().set_visible(False)
#i= ax.imshow(img_4_diameter[:,:,0],interpolation='nearest',aspect='auto')
#ax.set_title('Esimated diameter (Algorithm)')
#divider = make_axes_locatable(ax)
#cax = divider.new_vertical(size = '5%', pad = 0.01, pack_start = True)
#fig.add_axes(cax)
#cbar =plt.colorbar(i,orientation='horizontal', cax= cax) 
#cbar.ax.set_xlabel('\u03BCm')
#
#ax = fig.add_subplot(5, 3,12)
#ax.get_xaxis().set_visible(False)
#i =ax.imshow(new_predicted_img[3]*pixel_size*digital_zoom, aspect = 'auto',interpolation='nearest')
#ax.set_title('Predicted Diameter (Unet)')
#divider = make_axes_locatable(ax)
#cax = divider.new_vertical(size = '5%', pad = 0.01, pack_start = True)
#fig.add_axes(cax)
#cbar =plt.colorbar(i,orientation='horizontal', cax= cax) 
#cbar.ax.set_xlabel('\u03BCm')
#
#ax = fig.add_subplot(5, 3, 13)   
#ax.imshow(img_5,aspect='auto')
#ax.set_title('Raw Image Tortuous Vessles',size =10)
#
#ax = fig.add_subplot(5, 3, 14)
#ax.get_xaxis().set_visible(False)
#i= ax.imshow(img_5_diameter[:,:,0],interpolation='nearest',aspect='auto')
#ax.set_title('Esimated diameter (Algorithm)')
#divider = make_axes_locatable(ax)
#cax = divider.new_vertical(size = '5%', pad = 0.01, pack_start = True)
#fig.add_axes(cax)
#cbar =plt.colorbar(i,orientation='horizontal', cax= cax) 
#cbar.ax.set_xlabel('\u03BCm')
#
#ax = fig.add_subplot(5, 3,15)
#ax.get_xaxis().set_visible(False)
#i =ax.imshow(new_predicted_img[4]*pixel_size*digital_zoom, aspect = 'auto',interpolation='nearest')
#ax.set_title('Predicted Diameter (Unet)')
#divider = make_axes_locatable(ax)
#cax = divider.new_vertical(size = '5%', pad = 0.01, pack_start = True)
#fig.add_axes(cax)
#cbar =plt.colorbar(i,orientation='horizontal', cax= cax) 
#cbar.ax.set_xlabel('\u03BCm')








