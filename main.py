
from tkinter.constants import BOTTOM, E, LEFT, N, RIGHT, S, TOP, W
from tkinter.filedialog import askopenfile, askopenfilename
import tensorflow as tf
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from collections import Counter
import datetime

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from core.CV.diameter_shortest_distance import diameter_short
from patchify import patchify, unpatchify
from core.NN.Instance_semantic_segmentation_model import unet_multiclass
from core.CV.branch_detection import branch
from tkinter import DoubleVar, ttk
from ttkthemes import themed_tk as tk
import tkinter as tkin
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as mtick
from pandas import DataFrame
from tkinter import filedialog
import zipfile
import io
import seaborn as sns
import pandas as pd
from skimage.transform import resize
import numpy as np
import threading

    



class GUI():

    def __init__(self, master):

        #Begin variables and emty lists for saving and extracting data.
        self.output = []
        self.model = []
        self.master = master
        self.radius = 20
        self.tile_size = 1
        self.contrast_limit = 1
        self.sigma=6
        self.threshold_skl = -0.0051
        self.selection_field_elements = None
    
    
        self.scetch_model = []
        self.YLength = 15
        self.Thresh_thresh = 250
        self.mean = 180
        self.pixel_width = 5
        self.digital_zoom =1
        self.ROI_size = tkin.IntVar(self.master)
        self.selection_field = None
        self.selected_ROIS = None
        self.patches_def = None
        self.selected_image = tkin.IntVar(self.master)
        self.diameter_width_array = [0,2,5,7,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110]
        self.plotting_frame_array = []
        
        

        

        

        # Applying frames in widow for easy placement of graphs and buttons
        self.image_frame_input = ttk.Frame(self.master, width = 280, height = 280 ,style= "TFrame")
        self.image_frame_algorithm = ttk.Frame(self.master, width = 280, height = 280 ,style= "TFrame")
        #self.image_frame_tracking = ttk.Frame(self.master, width = 245, height = 240 ,style= "TFrame")
        self.image_frame_unet = ttk.Frame(self.master, width = 280, height = 280 ,style= "TFrame")
        self.image_frame_diameter_graph = ttk.Frame(self.master, width = 280, height = 280 ,style= "TFrame")
        #self.image_frame_velocity_graph = ttk.Frame(self.master, width = 245, height = 240 ,style= "TFrame")
        self.patch_selection_frame =ttk.Frame(self.master, width = 280, height=280 ,style= "TFrame")
        self.Bottom_frame = ttk.Frame(self.master, width = 560, height=200)
        self.run_frame = ttk.Frame(self.master, width= 280, height=280)
        self.maximum = 0

        # Positioning of frames in de window
        self.image_frame_input.grid(row=1,column =0,pady = 20,padx =20)
        self.image_frame_algorithm.grid(row=1,column =1,pady = 20,padx =20)
        #self.image_frame_tracking.grid(row=1,column =2)
        self.image_frame_unet.grid(row=1,column =2,pady = 20,padx =20)
        self.image_frame_diameter_graph.grid(row = 1,column =3,pady = 20,padx =20)
        #self.image_frame_velocity_graph.grid(row = 2,column =4)
        self.patch_selection_frame.grid(row = 2, column = 0,sticky =N)
        self.Bottom_frame.grid(row = 2, column = 1 , columnspan= 2,sticky = N)
        self.run_frame.grid(row=2,column= 3, pady =10,sticky =N)

        # Creating buttonlayout for visualition when selected
        self.style_run = ttk.Style(self.run_frame)
        self.style_run.layout('text.Horizontal.TProgressbar',
                     [('Horizontal.Progressbar.trough',
                       {'children': [('Horizontal.Progressbar.pbar',
                                      {'side': 'left', 'sticky': 'ns'})],
                        'sticky': 'nswe'}),
                      ('Horizontal.Progressbar.label', {'sticky': ''})])
                      # , lightcolor=None, bordercolo=None, darkcolor=None
        self.style_run.configure('text.Horizontal.TProgressbar', text='0 %')
        self.style_frame = ttk.Style()
        self.style_frame.configure("TFrame", background = 'white')

        self.style_text = ttk.Style()
        self.style_text.configure("TLabel", font = ('calibri', 13,'bold'), forground ='black', background= 'white' )

        self.style_checkbutton = ttk.Style()
        self.style_checkbutton.configure("TCheckbutton", background="wihte", forground = "white")



        self.style_button = ttk.Style()    
        self.style_button.configure('TButton', font =
               ('calibri', 11, 'bold'), foreground = 'green')
        self.style_button.configure('TButton', background='black')
        self.style_button_selected = ttk.Style()
        self.style_button_selected.configure('W.TButton', font =
               ('calibri', 11, 'bold'), foreground = 'red')
        self.style_button_selected.configure('W.TButton', background='black', borderwidth=1, focusthickness=3, focuscolor='none')   
            
        #Creating buttons, labels, enteries and graphs. All the buttons containing a function which will activivate an action.
        self.button_open_image = ttk.Button(self.image_frame_input,text='File',width = 20,command=self.open_image_file, style='TButton')
        self.button_algorithm = ttk.Button(self.image_frame_algorithm,text='Diameter Algorithm',width=20, command=self.Algorithm,style='TButton')
       # self.button_upload_model = ttk.Button(self.Sub_Top_frame,text='Upload pre-trained Unet model',width= 20, command=self.upload_model)
        #self.button_tracking = ttk.Button(self.image_frame_tracking,text='Tracking',width=20, command=self.Algorithm,style='TButton')
        #self.select_patch_image = ttk.Button(self.image_frame_input,text='Select ROI',width=30, command=self.select_roi,style='TButton')
        self.button_unet = ttk.Button(self.image_frame_unet,text='Unet',width=20, command=self.Unet, style = 'TButton')
        self.button_graph = ttk.Button(self.image_frame_diameter_graph,text='Graph',width=20, command=self.display_grpah, style = 'TButton')
        self.button_select_roi = ttk.Button(self.patch_selection_frame,text='Select ROI',width=20, command=self.select_roi, style = 'TButton') 
        self.button_activate_live_camera = ttk.Button(self.run_frame,text='Activate Camera',width=20, command=self.open_camera_frame, style = 'TButton')                                                         
        self.input_image_text = ttk.Label(self.image_frame_input, text='Input Image',style = "TLabel")
        self.algorithm_image_text = ttk.Label(self.image_frame_algorithm, text='Diameter Image',style = "TLabel")
        #self.tracking_image_text = ttk.Label(self.image_frame_tracking, text='Tracking Movie',style = "TLabel")
        #self.unet_image_text = ttk.Label(self.image_frame_unet, text='U-Net Image',style = "TLabel")
        #self.diameter_output_graph_text = ttk.Label(self.image_frame_diameter_graph, text='Diameter prediction',style = "TLabel")
        #self.velocity_output_graph_text = ttk.Label(self.image_frame_velocity_graph, text='Velocity Estimation',style = "TLabel")
        self.checkbox_1 = ttk.Checkbutton(self.patch_selection_frame,variable = self.selected_image, onvalue= 0, offvalue= -1, style = "TCheckbutton", command= self.show_variable)
        self.checkbox_2 = ttk.Checkbutton(self.patch_selection_frame,variable = self.selected_image, onvalue= 1, offvalue= -1, style = "TCheckbutton", command= self.show_variable)
        self.checkbox_3 = ttk.Checkbutton(self.patch_selection_frame,variable = self.selected_image, onvalue= 2, offvalue= -1, style = "TCheckbutton", command= self.show_variable)
        self.checkbox_4 = ttk.Checkbutton(self.patch_selection_frame,variable = self.selected_image, onvalue= 3, offvalue= -1, style = "TCheckbutton", command= self.show_variable)
        

        self.patch_slection_text = ttk.Label(self.patch_selection_frame, text = 'Selected Patch',style = "TLabel")

        


        #self.button_branch = ttk.Button(self.Top_frame,text='Branch structure',width=30,  command=self.Branch, style= 'TButton')
        
        self.button_export_data = ttk.Button(self.run_frame,text='Export Data',width= 20, command=self.Export_data)
        #self.button_select_data = ttk.Button(self.patch_slection_frame,text='Select patch',width= 20, command=self.select_patch)
        #self.button_delete_data = ttk.Button(self.patch_slection_frame,text='Delete all elements',width= 10, command=self.delete_selection)
        #self.button_alg_scetch = ttk.Button(self.image_frame_output,text='Scetch Algorithm',width= 13, command=self.scetch_model_alg, style= 'TButton')
        #self.button_unet_scetch = ttk.Button(self.image_frame_output,text='Scetch Unet',width= 10  , command=self.scetch_model_unet, style= 'TButton') 
        #self.button_branch_scetch = ttk.Button(self.image_frame_output,text='Scetch Branch',width= 10  , command=self.scetch_model_branch, style= 'TButton')               
        self.button_run = ttk.Button(self.run_frame,text='Run',width= 20, command=self.Run_app)  
        
        #self.output_image_text = ttk.Label(self.image_frame, text='Output Image',font=('calibri', 13, 'bold'))
        #Radius, threshold_skl, contrast cliplimit, tile size, sigma, filtersize,mean, thresholdthresh
        #self.data_frame_text = ttk.Label(self.data_frame, text = 'Output data',font=('calibri', 13, 'bold'))
        self.input_variables_header = ttk.Label(self.Bottom_frame, text = 'Insert variables',style = "TLabel")
        self.input_variables_radius_text = ttk.Label(self.Bottom_frame, text = 'Insert maximum radius')
        self.input_variables_radius= ttk.Entry(self.Bottom_frame,width=20)
        self.input_variables_tile_size= ttk.Entry(self.Bottom_frame,width=20)
        self.input_variables_tile_size_text = ttk.Label(self.Bottom_frame, text = 'Insert size of tile')
        self.input_variables_contrast_limit= ttk.Entry(self.Bottom_frame,width=20)
        self.input_variables_contrast_limit_text = ttk.Label(self.Bottom_frame, text = 'Insert contrast-cliplimit')
        self.input_variables_sigma= ttk.Entry(self.Bottom_frame,width=20)
        self.input_variables_sigma_text = ttk.Label(self.Bottom_frame, text = 'Insert sigma')
        self.input_variables_pixel_width= ttk.Entry(self.Bottom_frame,width=20)
        self.input_variables_pixel_width_text = ttk.Label(self.Bottom_frame, text = 'Insert pixel width h x b in \u03BCm')
        self.input_variables_digital_zoom_text = ttk.Label(self.Bottom_frame, text = 'Insert digital zoom of lens')
        self.input_variables_digital_zoom= ttk.Entry(self.Bottom_frame,width=20) 
        self.input_variables_YLength_text = ttk.Label(self.Bottom_frame, text = 'Insert matched filter size')
        self.input_variables_YLength= ttk.Entry(self.Bottom_frame,width=20) 
        self.input_variables_Thresh_text = ttk.Label(self.Bottom_frame, text = 'Insert threshold limit')
        self.input_variables_Thresh= ttk.Entry(self.Bottom_frame,width=20) 
        self.input_variables_mean_text = ttk.Label(self.Bottom_frame, text = 'Insert mean')
        self.input_variables_mean= ttk.Entry(self.Bottom_frame,width=20)
        #self.selection_field =ttk.Listbox(self.patch_slection_frame, selectmode=tk.SINGLE,width=10, height = 20)
        self.progres_bar = ttk.Progressbar(self.run_frame, orient='horizontal', length=200,mode= 'determinate',style = 'text.Horizontal.TProgressbar')
        

        #Positioning of the created objects within their selected frame.
        self.button_algorithm.grid(row=0)
        self.button_unet.grid(row=0)
        #self.button_branch.grid(row= 1, column = 2,padx =10, sticky=E)
        self.button_open_image.grid(row=0)
        #self.button_tracking.grid(row=0)
        self.button_graph.grid(row=0)
        self.button_select_roi.grid(row=0, column =0, columnspan= 4, pady = 10)
        self.checkbox_1.grid(row = 1, column =0)
        self.checkbox_2.grid(row = 1, column =2)
        self.checkbox_3.grid(row =2, column =0 )
        self.checkbox_4.grid(row =2, column = 2)

        #self.button_upload_model.grid(row = 1, column = 1, padx = 100 )
        #self.select_patch_image.grid(row=3,column=1)
        self.button_export_data.grid(row=4,column= 1)
        self.button_run.grid(row =1, column =1)
        self.button_activate_live_camera.grid(row =6 , column=1)
        #self.button_alg_scetch.grid(row = 3, column = 1)
        #self.button_unet_scetch.grid(row =3, column = 2)
        #self.button_branch_scetch.grid(row =3, column = 3)
        self.progres_bar.grid(row=2,column=1)
        self.make_scetch(self.image_frame_input,None,(2,2))
        self.make_scetch(self.image_frame_algorithm,None,(2,2))
        #self.make_scetch(self.image_frame_tracking,None,(2,2))
        self.make_scetch(self.image_frame_unet,None,(2,2))
        self.Make_chart(self.image_frame_diameter_graph,None,(2,2))
        #self.Make_chart(self.image_frame_velocity_graph,None,(2,2))
        self.scetch_selection(self.patch_selection_frame)

        #self.input_image_text.grid(row=3, pady = 0 ,sticky = N)
        #self.algorithm_image_text.grid(row=3, pady = 0 ,sticky = N)
        #self.tracking_image_text.grid(row=2, pady = 0 ,sticky = N)
        #self.unet_image_text.grid(row=3, pady = 0 ,sticky = N)
        #self.diameter_output_graph_text.grid(row=3, pady = 0 ,sticky = N)
        self.patch_slection_text.grid(row=3, columnspan=4, pady = 0 ,sticky = N)
        #self.velocity_output_graph_text.grid(row=2,  pady = 0 ,sticky = N)
        #self.output_image_text.grid(row=1, column = 2)
        
        #self.data_frame_text.grid(row=1, column = 2)
        self.input_variables_header.grid(row=1, columnspan = 5, pady= 15)  
        self.input_variables_pixel_width_text.grid(row = 3 ,column = 1,sticky=W, padx =2)
        self.input_variables_pixel_width.grid(row =3 ,column = 2,padx =2)
        self.input_variables_digital_zoom_text.grid(row=4,column = 1,sticky=W,padx =2)
        self.input_variables_digital_zoom.grid(row=4,column = 2,padx =2)
        
        
        self.input_variables_radius.grid(row =3, column= 4,padx =2)
        self.input_variables_radius_text.grid(row =3, column= 3,sticky=W,padx =2)
        #self.selection_field.grid(row=2, column=1)
        #Making empty scetch for begin layout
        
        
        
        
        
        self.input_variables_tile_size.grid(row = 5, column = 4,padx =2)
        self.input_variables_contrast_limit.grid(row=4, column = 4,padx =2)
        self.input_variables_sigma.grid(row=6, column =4,padx =2)
        self.input_variables_tile_size_text.grid(row=5, column =3,sticky=W,padx =2)
        self.input_variables_contrast_limit_text.grid(row=4, column =3,sticky=W,padx =2)
        self.input_variables_sigma_text.grid(row=6, column =3,sticky=W,padx =2)
        self.input_variables_YLength_text.grid(row=7, column =3,sticky=W,padx =2)
        self.input_variables_YLength.grid(row=7, column =4,padx =2)
        self.input_variables_Thresh_text.grid(row = 9, column =3,sticky=W,padx =2)
        self.input_variables_Thresh.grid(row = 9 ,column = 4,padx =2)
        self.input_variables_mean_text.grid(row = 8, column = 3,sticky=W,padx =2)
        self.input_variables_mean.grid(row = 8, column = 4,padx =2)


        #Begin variables for selecting patch frame
        self.return_index = True
        self.abort_value = None
        self.index_sets = []
        self.string_register = {}
        self.null_indices = []
        self.NULL_MARKER = '' or ''
        self.all_lines = []
    def show_variable(self):
        print(self.selected_image.get())
        if self.selected_image.get() != -1 and self.model.count('diameter algorithm') ==1: 
            image = self.patches_def[self.selected_ROIS_in_patch[self.selected_image.get()],:,:,0]
            self.make_scetch(self.image_frame_algorithm, image, (2,2))
            self.extract_data(self.image_frame_diameter_graph,image,(2,2))
        if self.selected_image.get() != -1 and self.model.count('Unet model') ==1: 
            image = self.patches_def[self.selected_ROIS_in_patch[self.selected_image.get()],:,:,1]
            self.make_scetch(self.image_frame_unet, image, (2,2))
            self.extract_data(self.image_frame_diameter_graph,image,(2,2))
    def show_variable_graph(self):
            print(self.selected_graph.get())
            if self.selected_graph.get() != -1 and self.scetch_model == 'Alg': 
                image = self.patches_def[self.selected_ROIS_in_patch[self.selected_graph.get()],:,:,0]
                self.extract_data(self.plotting_graph_frame,image,(4,4))
                self.make_scetch(self.plotting_diameter_frame,image,(4,4))
            if self.selected_graph.get() != -1 and self.scetch_model == 'Unet': 
                print("go")
                image = self.patches_def[self.selected_ROIS_in_patch[self.selected_graph.get()],:,:,1]
                self.extract_data(self.plotting_graph_frame,image,(4,4))
                self.make_scetch(self.plotting_diameter_frame,image,(4,4))
    def show_patch_size(self):
        print(self.patch_size.get())
        if self.patch_size.get() == 256:
            self.size_img = self.image.shape
            new_height = int(self.size_img[0]/self.patch_size.get())*self.patch_size.get()
            new_width = int(self.size_img[1]/self.patch_size.get())*self.patch_size.get()
            self.new_image = resize(self.image,(new_height,new_width,3),mode='constant',preserve_range='true') #resize image into patchable size
            self.new_image = np.array(self.new_image,  dtype = 'uint8')
            
            self.patches_image = patchify(self.new_image,(self.patch_size.get(),self.patch_size.get(),3), step =self.patch_size.get()) #patchify image
            self.selected_ROIS = np.ones((4,self.patch_size.get(),self.patch_size.get(),3),dtype = int)
            self.img_patch_normal  = np.ones(((self.patches_image.shape[0]*self.patches_image.shape[1]),self.patch_size.get(),self.patch_size.get(),3),dtype = int)
            a = 0
            if self.selection_field_elements is not None:
                self.index_sets = []
                self.string_register = {}
                self.null_indices = []
                self.all_lines = []
                self.selection_field_elements = None
                self.selection_field.delete(first= 0, last= 'end')
            self.img_label_list =[]
            for i in range(self.patches_image.shape[0]):
                    for j in range(self.patches_image.shape[1]):
                        image_patch = self.patches_image[i,j,0,:,:,:] #Taking patch
                        if len(self.img_label_list) < (self.patches_image.shape[0]*self.patches_image.shape[1]): #Checks if labels are made othewise every patch wil be labeld
                            self.img_patch_normal[a,:,:,:] = image_patch #save normal patch in array
                            self.img_label_list.append('Patch'+str(i)+'_'+str(j)+'_'+str(self.patch_size.get())+'.png') #saving label in list
                            a+=1
            if self.img_label_list.count('Total_img.png')!=1: #labeleing output image
                self.img_label_list.append('Total_img.png')
                    

        
        elif self.patch_size.get() == 512:
            self.size_img = self.image.shape
            new_height = int(self.size_img[0]/self.patch_size.get())*self.patch_size.get()
            new_width = int(self.size_img[1]/self.patch_size.get())*self.patch_size.get()
            self.new_image = resize(self.image,(new_height,new_width,3),mode='constant',preserve_range='true') #resize image into patchable size
            self.new_image = np.array(self.new_image,  dtype = 'uint8')

            self.patches_image = patchify(self.new_image,(self.patch_size.get(),self.patch_size.get(),3), step =self.patch_size.get()) #patchify image
            self.selected_ROIS = np.ones((4,self.patch_size.get(),self.patch_size.get(),3),dtype = int)
            self.img_patch_normal  = np.ones(((self.patches_image.shape[0]*self.patches_image.shape[1]),self.patch_size.get(),self.patch_size.get(),3),dtype = int)
            a = 0
            if self.selection_field_elements is not None:
                self.index_sets = []
                self.string_register = {}
                self.null_indices = []
                self.all_lines = []
                self.selection_field_elements = None
                self.selection_field.delete(first= 0, last= 'end')
            self.img_label_list =[]
            for i in range(self.patches_image.shape[0]):
                    for j in range(self.patches_image.shape[1]):
                        image_patch = self.patches_image[i,j,0,:,:,:] #Taking patch
                        if len(self.img_label_list) < (self.patches_image.shape[0]*self.patches_image.shape[1]): #Checks if labels are made othewise every patch wil be labeld
                            self.img_patch_normal[a,:,:,:] = image_patch #save normal patch in array
                            self.img_label_list.append('Patch'+str(i)+'_'+str(j)+'.png') #saving label in list
                            a+=1
            if self.img_label_list.count('Total_img.png')!=1: #labeleing output image
                self.img_label_list.append('Total_img.png')   
        self.selection_field_elements = self._parse_strings(self.img_label_list) #make selectio field elements
        self.selection_field.insert(0,*self.selection_field_elements) # insert labels in selection_field
        self.lastselection = None # the last Listbox item selected. 
        self.selection_field.bind('<<ListboxSelect>>', self._reselect) #Bind selection field with the reselect function
    def activate_camera(self):
        model =unet_multiclass(19, 512,512,3) #loading model
        model.load_weights('checkpoints/multiclass_model_512_test.h5') # adding wheights
        self.video = cv2.VideoCapture(0)
        self.diameter_width_array = [0,2,5,7,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110]
        self.close = 'no'
        pixel_width =1.43
        digital_zoom = 150
        while True:
            _, frame = self.video.read()
            if frame is None:
                break
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            images = patchify(frame,(512,512,3), step=512)
            frame = unpatchify(images,(512*images.shape[0],512*images.shape[1],3))
            print(images.shape)
            input_images = images.reshape(images.shape[0]*images.shape[1], 512,512,3)
            print(input_images.shape)
            prediction = (model.predict(input_images)) #Making prediction
            prediction = prediction.reshape(input_images.shape[0],512,512,19)
            print(prediction.shape)

         #reshaping output image
            predicted_img=np.argmax(prediction, axis=3)
            new_predicted_img= np.ones(predicted_img.shape,dtype = float)
            for k in range(predicted_img.shape[0]):    
                for i in range(predicted_img.shape[1]):
                    for j in range(predicted_img.shape[2]):
                        class_nr = predicted_img[k,i,j]
                        new_predicted_img[k,i,j] = self.diameter_width_array[class_nr]


            new_predicted_img = np.array(new_predicted_img)*pixel_width*digital_zoom
            new_predicted_img =np.round(new_predicted_img,1)
            reshape_prediction = new_predicted_img.reshape((images.shape[0],images.shape[1],512,512)) #reshaping for unpatchify
            segmented_video = unpatchify(reshape_prediction,(512*images.shape[0],512*images.shape[1])) #unpatchify for total image
            self.make_scetch(self.camera_frame,frame,(4,4))
            self.make_scetch(self.segmented_frame,segmented_video,(4,4))

            self.camera_window.update()
            if self.record == 'on':
                print('Recording')
                fig_input,ax1 = plt.subplots(nrows =2, ncols = 1,figsize = (8,8))
                i = ax1[1].imshow(segmented_video)
                ax1[1].get_xaxis().set_visible(False)     
                divider = make_axes_locatable(ax1[1])
                cax = divider.new_vertical(size = '5%', pad = 0.01, pack_start = True)
                fig_input.add_axes(cax)
                cbar =plt.colorbar(i,orientation='horizontal', cax= cax) 
                cbar.ax.set_xlabel('Diameter (\u03BCm)')
                i = ax1[0].imshow(frame)
                ax1[0].get_xaxis().set_visible(False)
                ax1[0].get_yaxis().set_visible(False)
                image = self.fig2data(fig_input)
                self.out.write(image)
                plt.close()

            elif self.close == 'yes':
                print('stop')
                break
            
        self.video.release()
        self.camera_window.destroy()

    def fig2data (self,fig):
        """
        @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
        @param fig a matplotlib figure
        @return a numpy 3D array of RGBA values
        """
        # draw the renderer
        fig.canvas.draw ( )
    
        # Get the RGBA buffer from the figure
        w,h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8 )
        buf.shape = ( w, h,4 )
    
        # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
        buf = np.roll(buf, 3, axis = 2 )
        buf = cv2.cvtColor(buf,cv2.COLOR_RGB2BGR)
        return buf    

    def open_image_file(self):
        #Function for opening file
        
        filepath = askopenfilename()
        

        if filepath:
            #Reading the image and plotting in the input graph
            image= cv2.imread(filepath)
            self.image = np.array(image)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.image = np.array(self.image,  dtype = 'uint8')
            self.make_scetch(self.image_frame_input,self.image, (2,2))
            self.make_scetch(self.image_frame_input,self.image, (2,2))
            
            
            
            
            
            
            
            
            
            
            
               
        
        return 
    def select_roi(self):
        
        
        self.selection_window = tk.ThemedTk()
        self.selection_window.get_themes()
        self.selection_window.set_theme("radiance")
        self.selection_window.geometry("700x600")
        self.selection_window.config(bg='white')
        self.selection_window.title('select ROI')

        self.style_checkbutton = ttk.Style(self.selection_window)
        self.style_checkbutton.configure("TCheckbutton", background="wihte", forground = "white")

        self.style_frame = ttk.Style(self.selection_window)
        self.style_frame.configure("TFrame", background = 'white')
        self.style_text = ttk.Style(self.selection_window)
        self.style_text. configure("TLabel", font = ('calibri', 13,'bold'), forground ='black', background= 'white' )

        selection_frame = ttk.Frame(self.selection_window, width = 400, height = 600 ,style= "TFrame")
        self.plotting_frame_1 = ttk.Frame(self.selection_window, width = 200, height = 300 ,style= "TFrame")
        self.plotting_frame_2 = ttk.Frame(self.selection_window, width = 200, height = 300 ,style= "TFrame")
        self.plotting_frame_3 = ttk.Frame(self.selection_window, width = 200, height = 300 ,style= "TFrame")
        self.plotting_frame_4 = ttk.Frame(self.selection_window, width = 200, height = 300 ,style= "TFrame")
        self.closing_frame = ttk.Frame(self.selection_window, width = 50, height= 50, style = "TFrame")
        self.plotting_frame_array.append(self.plotting_frame_1)
        self.plotting_frame_array.append(self.plotting_frame_2)
        self.plotting_frame_array.append(self.plotting_frame_3)
        self.plotting_frame_array.append(self.plotting_frame_4)
        
        self.selection_frame_number = 0
        self.patch_size = tkin.IntVar(self.selection_window)

        selection_frame.grid(row = 0, column=0, rowspan = 3)
        self.plotting_frame_1.grid(row = 0 , column=1)
        self.plotting_frame_2.grid(row = 0, column =2)
        self.plotting_frame_3.grid(row =1, column =1)
        self.plotting_frame_4.grid(row=1, column =2)
        self.closing_frame.grid(row =2,  column = 2)

        self.make_scetch(self.plotting_frame_1,None,(2,2))
        self.make_scetch(self.plotting_frame_2,None,(2,2))
        self.make_scetch(self.plotting_frame_3,None,(2,2))
        self.make_scetch(self.plotting_frame_4,None,(2,2))
        self.selection_field =tkin.Listbox(selection_frame, selectmode=tkin.SINGLE,width=10, height = 20)
        checkbox_5 = ttk.Checkbutton(selection_frame,variable = self.patch_size, onvalue= 256, offvalue= -1, style = "TCheckbutton", command= self.show_patch_size) 
        checkbox_6 = ttk.Checkbutton(selection_frame,variable = self.patch_size, onvalue= 512, offvalue= -1, style = "TCheckbutton", command= self.show_patch_size)
        self.button_select_data = ttk.Button(selection_frame,text='Select ROI',width= 20, command=self.select_roi_button)
        self.button_close_frame = ttk.Button(self.closing_frame,text='Done',width= 20, command=lambda: self.selection_window.destroy())
        selection_frame_text = ttk.Label(selection_frame, text='Select ROI',style = "TLabel")  
        checkbox_5_text = ttk.Label(selection_frame, text='256*256',style = "TLabel")  
        checkbox_6_text = ttk.Label(selection_frame, text='512*512',style = "TLabel")
        patch_size_text = ttk.Label(selection_frame, text='Selected ROI size',style = "TLabel")   
        plotting_frame_1_text = ttk.Label(self.plotting_frame_1, text='Selected ROI 1',style = "TLabel")
        plotting_frame_2_text = ttk.Label(self.plotting_frame_2, text='Selected ROI 2',style = "TLabel")
        plotting_frame_3_text = ttk.Label(self.plotting_frame_3, text='Selected ROI 3',style = "TLabel")
        plotting_frame_4_text = ttk.Label(self.plotting_frame_4, text='Selected ROI 4',style = "TLabel")
        self.button_close_frame.grid(row=0)
        self.button_select_data.grid(row =5, column =0, columnspan = 2)
        self.selection_field.grid(row =4, column = 0,columnspan = 2)
        selection_frame_text.grid(row=3,column = 0,columnspan = 2)
        checkbox_5.grid(row= 1, column = 0)
        checkbox_6.grid(row= 1, column =1)
        checkbox_5_text.grid(row = 2, column =0)
        checkbox_6_text.grid(row = 2, column =1)
        patch_size_text.grid(row =0 ,columnspan=2,column =0)


        
        plotting_frame_1_text.grid(row =0)
        plotting_frame_2_text.grid(row =0)
        plotting_frame_3_text.grid(row =0)
        plotting_frame_4_text.grid(row =0)
         #List for labeling data
        
        self.selected_ROIS_in_patch = []
         #array for storing input patches
    def open_camera_frame(self):
        self.camera_window = tk.ThemedTk()
        self.camera_window.get_themes()
        self.camera_window.set_theme("radiance")
        self.camera_window.geometry("900x500")
        self.camera_window.config(bg='white')
        self.camera_window.title('Live Camera')

        self.style_frame = ttk.Style(self.camera_window)
        self.style_frame.configure("TFrame", background = 'white')

        self.style_text = ttk.Style(self.camera_window)
        self.style_text. configure("TLabel", font = ('calibri', 13,'bold'), forground ='black', background= 'white' )

        self.style_checkbutton = ttk.Style(self.camera_window)
        self.style_checkbutton.configure("TCheckbutton", background="wihte", forground = "white")
        self.style_text = ttk.Style(self.camera_window)
        self.style_text.configure("TLabel", font = ('calibri', 13,'bold'), forground ='black', background= 'white' )
        self.style_checkbutton = ttk.Style(self.camera_window)
        self.style_checkbutton.configure("TCheckbutton", background="wihte", forground = "white")
        self.style_button = ttk.Style(self.camera_window)    
        self.style_button.configure('TButton', font =
               ('calibri', 11, 'bold'), foreground = 'green')
        self.style_button.configure('TButton', background='black')
        self.style_button_selected = ttk.Style(self.camera_window)
        self.style_button_selected.configure('W.TButton', font =
               ('calibri', 11, 'bold'), foreground = 'red', text= 'Recording...')
        self.style_button_selected.configure('W.TButton', background='black', borderwidth=1, focusthickness=3, focuscolor='none')   


        self.camera_frame=ttk.Frame(self.camera_window, width = 400, height =500 ,style= "TFrame")
        self.segmented_frame = ttk.Frame(self.camera_window, width = 400, height =500 ,style= "TFrame")
        self.button_frame = ttk.Frame(self.camera_window, width = 800, height =10 ,style= "TFrame")
        self.button_close_camera_frame = ttk.Button(self.button_frame,text='Done',width= 20, command=self.camera_window_destroy, style = 'TButton')
        self.button_capture_video = ttk.Button(self.button_frame,text='Record',width= 20, command=  self.record_video,style = 'TButton')
        self.camera_frame.grid(row=0,column=0)
        self.segmented_frame.grid(row=0, column =1)
        self.button_frame.grid(row=1, columnspan=2, column = 0)
        self.button_close_camera_frame.grid(row=0,column=1, sticky=E)
        self.button_capture_video.grid(row=0, column=0, sticky=W)
        self.record = 'off'
        self.activate_camera()

    def camera_window_destroy(self):
        if self.close == 'no':
            self.close =  'yes'
            self.out.release()
            print(self.close)

        
    def record_video(self):
        if self.record == 'off':
            self.record = 'on'
            print(self.record)
            self.button_capture_video.config(style = 'W.TButton')
            width = 800
            height = 800
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use the lower case
            self.out = cv2.VideoWriter('video_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") +'_.mp4', fourcc, 5, (width, height))
        elif self.record =='on':
            self.record = 'off'
            print(self.record) 
            self.out.release()
            self.button_capture_video.config(style = 'TButton')
        return
        
    def upload_model(self):
        #options for uploading own trained Unet weigthes
        filepath = askopenfilename(filetypes=[("h5 - files", "*.h5")])

        if filepath:
            self.unet_model = filepath
        return
    def scetch_model_alg(self):
        #Changing scetch model into algorithm. This chooses which output wil be plotted
        if self.scetch_model == 'Alg':
            self.scetch_model = []
            self.button_alg_scetch.config(style = 'TButton')
        else:
            self.scetch_model = 'Alg'
            self.button_alg_scetch.config(style = 'W.TButton')
            self.show_variable_graph()
            self.button_unet_scetch.config(style = 'TButton')
            #self.button_branch_scetch.config(style = 'TButton')
    def scetch_model_unet(self):
        #Changing scetch model into Unet. This chooses which output wil be plotted
        if self.scetch_model == 'Unet':
            self.scetch_model = []
            self.button_unet_scetch.config(style = 'TButton')
        else:   
            self.scetch_model ='Unet'
            self.button_unet_scetch.config(style = 'W.TButton') 
            self.show_variable_graph()
            #self.button_branch_scetch.config(style = 'TButton')
            self.button_alg_scetch.config(style = 'TButton')
    def scetch_model_branch(self):
        #Changing scetch model into Branch. This chooses which output wil be plotted
        if self.scetch_model == 'Branch':
            self.scetch_model = []
            self.button_branch_scetch.config(style = 'TButton')
        else:   
            self.scetch_model = 'Branch'
            self.button_branch_scetch.config(style = 'W.TButton')
            self.button_alg_scetch.config(style = 'TButton')
            self.button_unet_scetch.config(style = 'TButton') 
    def display_grpah(self):
        self.graph_window = tk.ThemedTk()
        self.graph_window.get_themes()
        self.graph_window.set_theme("radiance")
        self.graph_window.geometry("1200x500")
        self.graph_window.config(bg='white')
        self.graph_window.title('View Graph')
        
        self.style_frame = ttk.Style(self.graph_window)
        self.style_frame.configure("TFrame", background = 'white')
        
        self.style_text = ttk.Style(self.graph_window)
        self.style_text. configure("TLabel", font = ('calibri', 13,'bold'), forground ='black', background= 'white' )
        
        self.style_checkbutton = ttk.Style(self.graph_window)
        self.style_checkbutton.configure("TCheckbutton", background="wihte", forground = "white")

        self.style_text = ttk.Style(self.graph_window)
        self.style_text.configure("TLabel", font = ('calibri', 13,'bold'), forground ='black', background= 'white' )
        self.style_checkbutton = ttk.Style(self.graph_window)
        self.style_checkbutton.configure("TCheckbutton", background="wihte", forground = "white")
        self.style_button = ttk.Style(self.graph_window)    
        self.style_button.configure('TButton', font =
               ('calibri', 11, 'bold'), foreground = 'green')
        self.style_button.configure('TButton', background='black')
        self.style_button_selected = ttk.Style(self.graph_window)
        self.style_button_selected.configure('W.TButton', font =
               ('calibri', 11, 'bold'), foreground = 'red')
        self.style_button_selected.configure('W.TButton', background='black', borderwidth=1, focusthickness=3, focuscolor='none')   

        self.selection_frame_graph = ttk.Frame(self.graph_window, width = 400, height =500 ,style= "TFrame")
        self.plotting_graph_frame = ttk.Frame(self.graph_window, width = 400, height = 500 ,style= "TFrame")
        self.plotting_diameter_frame = ttk.Frame(self.graph_window, width = 400, height = 500 ,style= "TFrame")
        self.closing_frame_graph = ttk.Frame(self.graph_window, width = 400, height = 20, style= "TFrame" )
        self.scetching_frame = ttk.Frame(self.graph_window, width = 400, height = 20, style= "TFrame" )
        self.selected_graph = tkin.IntVar(self.graph_window)
        self.selection_frame_graph.grid(column=0, row =0, padx=5)
        self.plotting_graph_frame.grid(column=2, row =0)
        self.plotting_diameter_frame.grid(column = 1, row =0)
        self.closing_frame_graph.grid(row =1, column =2)
        self.scetching_frame.grid(row =1,column = 1)
        
        selection_frame_text =ttk.Label(self.selection_frame_graph,text = "Selected ROI", style="TLabel")
        diameter_plotting_frame_text = ttk.Label(self.plotting_diameter_frame,text = "Diameter Estimation", style="TLabel")
        graph_X_text =  ttk.Label(self.plotting_graph_frame,text = "Diameter (\u03BCm)", style="TLabel")
        self.checkbox_graph_1 = ttk.Checkbutton(self.selection_frame_graph,variable = self.selected_graph, onvalue= 0, offvalue= -1, style = "TCheckbutton", command= self.show_variable_graph)
        self.checkbox_graph_2 = ttk.Checkbutton(self.selection_frame_graph,variable = self.selected_graph, onvalue= 1, offvalue= -1, style = "TCheckbutton", command= self.show_variable_graph)
        self.checkbox_graph_3 = ttk.Checkbutton(self.selection_frame_graph,variable = self.selected_graph, onvalue= 2, offvalue= -1, style = "TCheckbutton", command= self.show_variable_graph)
        self.checkbox_graph_4 = ttk.Checkbutton(self.selection_frame_graph,variable = self.selected_graph, onvalue= 3, offvalue= -1, style = "TCheckbutton", command= self.show_variable_graph)
        self.button_close_graph_frame = ttk.Button(self.closing_frame_graph,text='Done',width= 20, command=lambda: self.graph_window.destroy())   
        self.button_alg_scetch = ttk.Button(self.scetching_frame, text ="Scetch Algorithm", width = 20, style = "TButton",command=self.scetch_model_alg)
        self.button_unet_scetch = ttk.Button(self.scetching_frame, text ="Scetch Unet", width = 20,style = "TButton",command = self.scetch_model_unet)
        

        selection_frame_text.grid(row = 3, columnspan = 4)
        diameter_plotting_frame_text.grid(row =0,sticky = S)
        self.button_alg_scetch.grid(column =0,padx =5,row=0)
        self.button_unet_scetch.grid(column=1, padx =5,row=0)


        self.button_close_graph_frame.grid()
        self.checkbox_graph_1.grid(row = 1, column =0)
        self.checkbox_graph_2.grid(row = 1, column =2)
        self.checkbox_graph_3.grid(row =2, column =0)
        self.checkbox_graph_4.grid(row =2, column = 2)
        self.scetch_selection(self.selection_frame_graph)
        self.make_scetch(self.plotting_diameter_frame,None,(4,4))
        self.Make_chart(self.plotting_graph_frame,None,(4,4))
        graph_X_text.grid(row=2,sticky = N)
    def scetch_selection(self,location):
        self.fig_input_1,ax1 = plt.subplots(figsize = (1,1))
        self.fig_input_2,ax2 = plt.subplots(figsize = (1,1))
        self.fig_input_3,ax3 = plt.subplots(figsize = (1,1))
        self.fig_input_4,ax4 = plt.subplots(figsize = (1,1))
        ax1.get_xaxis().set_visible(False)
        ax2.get_xaxis().set_visible(False)
        ax3.get_xaxis().set_visible(False)
        ax4.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        ax3.get_yaxis().set_visible(False)
        ax4.get_yaxis().set_visible(False)
        if self.selected_ROIS is not None:
            print('nice')
            ax1.imshow(self.selected_ROIS[0])
            ax2.imshow(self.selected_ROIS[1])
            ax3.imshow(self.selected_ROIS[2])
            ax4.imshow(self.selected_ROIS[3])
        canvas_1= FigureCanvasTkAgg(self.fig_input_1,location) #Makes the pyplot figure readable for Tkinter
        canvas_2= FigureCanvasTkAgg(self.fig_input_2,location)
        canvas_3= FigureCanvasTkAgg(self.fig_input_3,location)
        canvas_4= FigureCanvasTkAgg(self.fig_input_4,location)
        canvas_1.draw() #Applys it
        canvas_2.draw()
        canvas_3.draw()
        canvas_4.draw()
        canvas_1.get_tk_widget().grid(row = 1, column= 1)
        canvas_2.get_tk_widget().grid(row = 1, column =3)
        canvas_3.get_tk_widget().grid(row = 2, column= 1)
        canvas_4.get_tk_widget().grid(row = 2, column =3) #Determine the position
        plt.close() #Close the pyplot figure so the memory won't be overloaded
    def make_scetch(self,location,image,size):
        self.fig_input,ax1 = plt.subplots(figsize = size)
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
         #Creating figure in pyplot
        if image is not None: #check if there is a uploaded image
            image = np.array(image)
            i = ax1.imshow(image)
            unique_value =  np.unique(image).tolist() #Create a list with al the unique values of the image
            my_list = [0,5,255] #Unique structure of a picture loaded with Branch
            if location is not (self.image_frame_input):
                if location not in self.plotting_frame_array:   #Make sure that only the Unet and ALgorithm output get an colorbar   
                        ax1.get_xaxis().set_visible(False)     
                        divider = make_axes_locatable(ax1)
                        cax = divider.new_vertical(size = '5%', pad = 0.01, pack_start = True)
                        self.fig_input.add_axes(cax)
                        cbar =plt.colorbar(i,orientation='horizontal', cax= cax) 
                        cbar.ax.set_xlabel('Diameter (\u03BCm)')
            if unique_value == my_list: #Check if the piocture is an output from the branch calculation and makes sure the splitpoints will be marked
                img_split = np.where(image==5)
                for k in range(0,len(img_split[0])):
                    y1,x1 = img_split[0][k], img_split[1][k]
                    ax1.plot(x1,y1,'bo')

        canvas= FigureCanvasTkAgg(self.fig_input,location) #Makes the pyplot figure readable for Tkinter
        canvas.draw() #Applys it
        canvas.get_tk_widget().grid(row = 1,column =0,ipadx =20,ipady =20 , sticky = N) #Determine the position
        plt.close() #Close the pyplot figure so the memory won't be overloaded
        return
    def Make_chart(self,location,dataframe,size): # Function for plotting extracted data
        self.fig_input_2, ax2 = plt.subplots(figsize = size)
        
        if dataframe is not None:
            if 'Diameter (\u03BCm)'in dataframe:
                color = sns.color_palette("viridis", 1+len(np.array(dataframe['Diameter (\u03BCm)'])))
                ax = sns.barplot(x='Diameter (\u03BCm)', y='Part of total vessel area', data=dataframe,ax=ax2,palette=color[1:])
                ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

                #Plotting diameter if dataframe contains a column diameter
            else: 
                sns.barplot(x='Curvature vessel', y='Numerical', data=dataframe,ax=ax2)     #Plotting split-points if dataframe contains not a column diameter

            
        
        canvas= FigureCanvasTkAgg(self.fig_input_2,location) #Makes the pyplot figure readable for Tkinter
        canvas.draw()
        canvas.get_tk_widget().grid(row = 1,column =0,ipadx =20,ipady = 20, sticky = N)
        plt.close()   
        return
    def close_window(self):
        self.selection_window.destroy()
        self.graph_window.destroy()
    def delete_selection(self): #Function which clears all values in the lists
        if self.selection_field_elements:
            self.selection_field.delete(first= 0, last= 'end')
        self.index_sets = []
        self.string_register = {}
        self.null_indices = []
        self.all_lines = []
        self.selection_field_elements = []
        self.img_label_list_algorithm = []
        self.img_label_list_unet = []
        self.img_label_list_branch = []
    
    def select_roi_button(self, event=None):
        if self.lastselection is None:
            self.result = None
        if self.lastselection  == len(self.img_label_list)-1: #Selecting total output image of algoritm
            self.normal = self.image   #Selecting input image
        else:
            self.normal = self.img_patch_normal[self.lastselection,:,:,:] #Selecting normal patch image of output image
        self.make_scetch(self.plotting_frame_array[self.selection_frame_number], self.normal,(2,2))
        self.selected_ROIS[self.selection_frame_number] = self.normal
        self.selected_ROIS_in_patch.append(self.lastselection)
        
        if self.selection_frame_number == 3:
            print('done')
            self.scetch_selection(self.patch_selection_frame)
            self.selection_frame_number = -1
        self.selection_frame_number += 1
    def select_patch(self, event=None): #Function for selecting data out of patch-frame
        if self.lastselection is None: # Nothing selected.
            self.result = None
        else: # Store the result in the `result`
            if self.scetch_model == 'Alg': #Checks which sketching method is selected
                if self.lastselection  == len(self.img_label_list)-1: #Selecting total output image of algoritm
                    self.result = self.total_def[:,:,0] #Selecting total output image of algoritm
                    self.normal = self.image   #Selecting input image
                else:
                    self.result = self.patches_def[self.lastselection,:,:,0] #Selecting patch output image of algorithm
                    self.normal = self.img_patch_normal[self.lastselection,:,:,:] #Selecting normal patch image of output image
                
                self.make_scetch(self.image_frame_output,self.result) #giving ouput image to scetching function
                self.make_scetch(self.image_frame_input,self.normal)  #giving input image to scetching function
                ignore =self.extract_data(self.data_frame,self.result) #Extract data of output and scetching in dataframe
            elif self.scetch_model == 'Unet':
                if self.lastselection  == len(self.img_label_list)-1: #Selecting total output image of unet
                    self.result = self.total_def[:,:,1] #Selecting total output image of unet
                    self.normal = self.image   #Selecting input image
                else:
                    self.result = self.patches_def[self.lastselection,:,:,1] #Selecting patch output image of unet
                    self.normal = self.img_patch_normal[self.lastselection,:,:,:] #Selecting normal patch image of output image

                self.make_scetch(self.image_frame_output,self.result)       #giving ouput image to scetching function
                self.make_scetch(self.image_frame_input, self.normal)       #giving input image to scetching function
                ignore =self.extract_data(self.data_frame,self.result) #Extract data of output and scetching in dataframe
                
            elif self.scetch_model == 'Branch':
                if self.lastselection  == len(self.img_label_list)-1: # Checks if it is  the last selection of list selection 
                    self.result = self.total_def[:,:,2] #Selecting total output image of branch
                    self.normal = self.image   #Selecting input image
                else:
                    self.result = self.patches_def[self.lastselection,:,:,2] #Selecting patch output image of branch
                    self.normal = self.img_patch_normal[self.lastselection,:,:,:] #Selecting normal patch image of output image

                self.make_scetch(self.image_frame_output,self.result)       #giving ouput image to scetching function
                self.make_scetch(self.image_frame_input, self.normal)       #giving input image to scetching function
                ignore = self.extract_data(self.data_frame,self.result)     #Extract data of output and scetching in dataframe
        

        return
    def extract_data(self,frame,image,size):
        string_list = ['split-point'] 
        part_list = []
        if len(np.unique(image)) != 3: #Check if the input image is a branch structure
            image_list = image.flatten().tolist()
            number_dict = Counter(image_list).keys()
            amount_dict = Counter(image_list).values()
            number_list = [*number_dict]
            amount_list = [*amount_dict]
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

            self.data_diameter = {'Diameter (\u03BCm)': np.array(number_list[1:]), 'Amount of Pixels' : np.array(amount_list[1:]), 'Part of total vessel area' : np.array(part_list)} #Making dictionary
            df1 = DataFrame(self.data_diameter,columns=['Diameter (\u03BCm)','Amount of Pixels','Part of total vessel area']) #Making dataframe of dictionary
        else:
            image = image.reshape(-1,1) 
            image_list = image.tolist()
            array_number = np.unique(image)
            amount_list =  []
            string_list = ['split-point']

            amount = image_list.count(array_number[1]) #COunts the pixels which where classified as splitpoints.
            amount_list.append(amount)

            self.data_diameter = {'Curvature vessel': np.array(string_list), 'Numerical' : np.array(amount_list)} #Making dictionary
            df1 = DataFrame(self.data_diameter,columns=['Curvature vessel','Numerical']) #Making dataframe of dictionary
        if frame is not None and size is not None:       
            self.Make_chart(frame,df1,size) #Plot the data into extratcted data frame
        return np.array(number_list[1:]),  np.array(part_list),  np.array(string_list), np.array(amount_list) # Outpus of all the existing data. 
    def _parse_strings(self, string_list):
        #Accepts a list of strings and breaks each string into a series of lines, logs the sets, and stores them in the item_roster and string_register attributes.
        index_sets = self.index_sets
        register = self.string_register
        REG_INDEX = self.return_index # register the item's index in the string list if truey.

         # holds all strings after lines are split. used for Lb elements.
        line_number = 0
        for n, item in enumerate(string_list): # string_register: `n` is saved if REG_INDEX, else `item`
            if not item: # null strings or falsey elemetns add a blank space. useful organizationally.
                self.null_indices.append(line_number) # record the blank position
                self.all_lines.append(self.NULL_MARKER) # add the divider text (or '')
                index_sets.append(None) # add an index value for the gap

                line_number += 1 # increment the line number.
                continue
            
            lines = item.splitlines()

            self.all_lines.extend(lines) # add the divided string to the string stack
            register[line_number] = n if REG_INDEX else item
                # ^ Saves this item keyed to the first Listbox element it's associated with.
                #  Register the string_list index if REG_INDEX, otherwise register the unbroken string.
                #  Dividers are not registered.

            qty = len(lines)
            if qty == 1: # single line item..
                index_sets.append((line_number, line_number))
            else: # multiple lines in this item..
                element_range = line_number, line_number + qty - 1 # the range of Listbox indices..
                index_sets.extend([element_range] * qty) # ..one for each line in the Listbox.

            line_number += qty # increment the line number.
            
        return self.all_lines   
    def _reselect(self, event=None):
        "Called whenever the Listbox's selection changes."
        selection = self.selection_field.curselection() # Get the new selection data.
        if not selection: # if there is nothing selected, do nothing.
            self.lastselection = None
            return

        if selection[0] in self.null_indices: # selected a divider..
            self._clear() # ..so clear the current selection.
            return
    
        lines_st, lines_ed = self.index_sets[selection[0]]
            # ^ Get the string block associated with the current selection.
        
        if lines_st == self.lastselection:
            self._clear()
            return # If the new set is the same as the old one, clear it.

        self.selection_field.selection_set(lines_st, lines_ed) # select all relevant lines
        self.lastselection = lines_st # remember what you selected last
        if self.lastselection  == len(self.img_label_list)-1: #Selecting total output of images
            self.normal = self.image   #Selecting input image
        else:
            self.normal = self.img_patch_normal[self.lastselection,:,:,:] #Selecting normal patch image of output image

        self.make_scetch(self.plotting_frame_array[self.selection_frame_number],self.normal, (2,2))
    def _clear(self, event=None):
        "Clears the current selection."
        selection = self.selection_field.curselection() # Get the currently selected item(s).
        if not selection: # Nothing to clear!
            return
        self.selection_field.selection_clear(selection[0], selection[-1]) # deselect..
        self.lastselection = None # ..and remember that you deselected!
        return
    def Algorithm(self): #Select algorithm model if algorithm button is pressed
        if self.model.count( 'diameter algorithm')==1:
            self.model.remove('diameter algorithm')
            self.button_algorithm.config(style = 'TButton')
            print(self.model)
        else:
            self.model.append( 'diameter algorithm')
            self.button_algorithm.config(style = 'W.TButton')
            print(self.model)
        
        return       
    def Unet(self): #Select unet model if unet button is pressed
        if self.model.count('Unet model')==1:
            self.model.remove('Unet model')
            self.button_unet.config(style = 'TButton')
            print(self.model)
        else:
            self.model.append('Unet model')
            self.button_unet.config(style = 'W.TButton')
            print(self.model)
        
        return      
    #def Branch(self):
        #Select branch model if branch button is pressed
        if self.model.count('Branch structure') ==1:
            self.model.remove('Branch structure')
            self.button_branch.config(style = 'TButton')
            print(self.model)
        else:
            self.model.append('Branch structure')
            self.button_branch.config(style = 'W.TButton')
            
            
            print(self.model)
        retur
    def Update_progres(self, value):  #Function for progress bar 
        self.progres_bar['value'] = value
    def Run_app(self): #Function which patchifys image and runs diffrent models
        self.style_run.configure('text.Horizontal.TProgressbar', text='0 %') #Setting progres bar to zero

        
        if self.input_variables_radius.get() != '':
            self.radius = float(self.input_variables_radius.get())
        if self.input_variables_tile_size.get() != '':
            self.tile_size =int(self.input_variables_tile_size.get())
        if  self.input_variables_sigma.get() != '':
            self.sigma =  float(self.input_variables_sigma.get())
        if  self.input_variables_contrast_limit.get()!= '':
            self.contrast_limit = float(self.input_variables_contrast_limit.get())
        if  self.input_variables_pixel_width.get()!= '':
            self.pixel_width = float(self.input_variables_pixel_width.get())
        if  self.input_variables_digital_zoom.get()!= '':
            self.digital_zoom = float(self.input_variables_digital_zoom.get())
        if  self.input_variables_YLength.get()!= '':
            self.YLength = float(self.input_variables_YLength.get())
        if  self.input_variables_Thresh.get()!= '':
            self.Thresh_thresh = float(self.input_variables_Thresh.get())
        if  self.input_variables_mean.get()!= '':
            self.mean = float(self.input_variables_mean.get())
        #Load all the variables if inserted in enteries and emprying all the lists

        if self.model: #Check if a model is selected
            self.img_label_list =[] #List for labeling data
            self.patches_def =  np.ones(((self.patches_image.shape[0]*self.patches_image.shape[1]),self.patch_size.get(),self.patch_size.get(),3),dtype = float)  # array for storing output patches 
            self.img_patch_normal  = np.ones(((self.patches_image.shape[0]*self.patches_image.shape[1]),self.patch_size.get(),self.patch_size.get(),3),dtype = int)     #array for storing input patches
            self.total_def = np.ones((self.new_image.shape[0],self.new_image.shape[1],3),dtype = float) #Array for storing output total images
            
            a=0
            b=0
            self.maximum = (self.patches_image.shape[0]*self.patches_image.shape[1])*(self.model.count('diameter algorithm')+self.model.count('Unet model')+self.model.count('Branch structure')) #Counting maximum value of progress bar
            print(self.maximum)
            self.progres_bar.config(mode='determinate', maximum=self.maximum, value=a) #Insert maximmum and setting active value on zero


            if self.model.count('diameter algorithm') ==1 and self.selected_ROIS is not None: #Checking if algorithm model is selected and if there is an input image
                def_img_diameter = np.ones((self.patches_image.shape[0],self.patches_image.shape[1],self.patch_size.get(),self.patch_size.get()),dtype = float)   #Making array which is in the form to unpatchify it back to original image
                
                
                
                for i in range(self.patches_image.shape[0]):
                    for j in range(self.patches_image.shape[1]):
                        image_patch = self.patches_image[i,j,0,:,:,:] #Taking patch
                        if len(self.img_label_list) < (self.patches_image.shape[0]*self.patches_image.shape[1]): #Checks if labels are made othewise every patch wil be labeld
                            self.img_patch_normal[a,:,:,:] = image_patch #save normal patch in array
                            self.img_label_list.append('Patch'+str(i)+'_'+str(j)+'_'+str(self.patch_size.get())+'.png') #saving label in list
                        self.diameter = diameter_short(image_patch,self.radius,self.threshold_skl,self.contrast_limit,self.tile_size,self.sigma,self.YLength,self.mean,self.Thresh_thresh)
                        self.real_diameter =self.diameter*self.pixel_width*self.digital_zoom
                        self.real_diameter =np.round(self.real_diameter,1) #deteming diameter map with diameter functon
                        def_img_diameter[i,j,:,:] = self.real_diameter #saving output image in array for unpatchify
                        print('done')
                        self.patches_def[b,:,:,0]=self.real_diameter #saving output image in array
                        b+= 1 #Update number for array position
                        a+=1  #Update number for porgressbar positioon
                        self.progres_bar.after(500, self.Update_progres(a))
                        self.run_frame.update()
                        self.style_run.configure('text.Horizontal.TProgressbar',
                            text='{:g} %'.format(((a/self.maximum)*100))) 
                self.total_def[:,:,0] = unpatchify(def_img_diameter, (self.patch_size.get()*self.patches_image.shape[0],self.patch_size.get()*self.patches_image.shape[1])) #Creating total output image out of patche
                if self.img_label_list.count('Total_img.png')!=1: #labeleing output image
                    self.img_label_list.append('Total_img.png')
                if self.selected_image.get() != -1:
                    image = self.patches_def[self.selected_ROIS_in_patch[self.selected_image.get()],:,:,0]
                    self.make_scetch(self.image_frame_algorithm, image,(2,2))
                    self.extract_data(self.image_frame_diameter_graph,image,(2,2))

                    

                   
            if self.model.count('Unet model') ==1 and self.image is not None:
                if len(self.img_label_list)  < (self.patches_image.shape[0]*self.patches_image.shape[1]):
                    for i in range(self.patches_image.shape[0]):
                        for j in range(self.patches_image.shape[1]):
                        
                            image_patch = self.patches_image[i,j,0,:,:,:]
                            self.img_patch_normal[a,:,:,:] = image_patch
                            self.img_label_list.append('Patch'+str(i)+'_'+str(j)+'_'+str(self.patch_size.get())+'.png')
                            a+=1 
                            self.progres_bar.after(500, self.Update_progres(a))
                            self.run_frame.update()
                            self.style_run.configure('text.Horizontal.TProgressbar',
                                text='{:g} %'.format(((a/self.maximum)*100)))
                if len(self.img_label_list)-1  == (self.patches_image.shape[0]*self.patches_image.shape[1]):
                    a += (self.patches_image.shape[0]*self.patches_image.shape[1])
                    self.progres_bar.after(500, self.Update_progres(a))
                    self.run_frame.update()
                    self.style_run.configure('text.Horizontal.TProgressbar',
                        text='{:g} %'.format(((a/self.maximum)*100)))
                    print(a)
                    
                    
                self.img_patch_normal_unet = self.patches_image.reshape((self.patches_image.shape[0]*self.patches_image.shape[1],self.patch_size.get(),self.patch_size.get(),3)) #reshape patches in a formating fitting for unet model
            
                model =unet_multiclass(19, self.patch_size.get(), self.patch_size.get(),3) #loading model
                model.load_weights('checkpoints/multiclass_model_'+str(self.patch_size.get())+'_test.h5') # adding wheights
                prediction = (model.predict(self.img_patch_normal)) #Making prediction
                prediction = prediction.reshape(self.img_patch_normal.shape[0],self.patch_size.get(),self.patch_size.get(),19)
                
                 #reshaping output image
                self.predicted_img=np.argmax(prediction, axis=3)
                self.new_predicted_img= np.ones(self.predicted_img.shape,dtype = float)
                print(self.new_predicted_img.shape)
                for k in range(self.predicted_img.shape[0]):    
                    for i in range(self.predicted_img.shape[1]):
                        for j in range(self.predicted_img.shape[2]):
                            class_nr = self.predicted_img[k,i,j]
                            self.new_predicted_img[k,i,j] = self.diameter_width_array[class_nr]

                #detemning output image
                self.new_predicted_img = np.array(self.new_predicted_img)*self.pixel_width*self.digital_zoom
                self.new_predicted_img =np.round(self.new_predicted_img,1)
                self.patches_def[:,:,:,1] =self.new_predicted_img #saving ouput patches
                
                reshape_prediction = self.new_predicted_img.reshape((self.patches_image.shape[0],self.patches_image.shape[1],self.patch_size.get(),self.patch_size.get())) #reshaping for unpatchify
                self.total_def[:,:,1] = unpatchify(reshape_prediction,(self.patch_size.get()*self.patches_image.shape[0],self.patch_size.get()*self.patches_image.shape[1])) #unpatchify for total image
                if self.img_label_list.count('Total_img.png')!=1:
                    self.img_label_list.append('Total_img.png')
                if self.selected_image.get() != -1:
                    image = self.patches_def[self.selected_ROIS_in_patch[self.selected_image.get()],:,:,1]
                    self.make_scetch(self.image_frame_unet, image,(2,2))
                    self.extract_data(self.image_frame_diameter_graph,image,(2,2))
    
            #if self.model.count('Branch structure') ==1 and self.image is not None:
            #    def_img_branch = np.ones((self.patches_image.shape[0],self.patches_image.shape[1],self.patch_size.get(),512),dtype = int) 
            #    b=0
            #    for i in range(self.patches_image.shape[0]):
            #        for j in range(self.patches_image.shape[1]):
            #            image_patch = self.patches_image[i,j,0,:,:,:]
            #            if len(self.img_label_list) < (self.patches_image.shape[0]*self.patches_image.shape[1]):
            #                self.img_patch_normal[a,:,:,:] = image_patch
            #                self.img_label_list.append('Patch'+str(i)+'_'+str(j)+'.png')
            #            
            #            print(image_patch.shape)
            #            self.branch = branch(image_patch,self.threshold_skl,self.contrast_limit,self.filter_size,self.sigma,self.YLength,  self.mean,self.Thresh_thresh)
            #            def_img_branch[i,j,:,:] = self.branch
            #            
            #            self.patches_def[b,:,:,2]=self.branch
            #            b+=1
            #            a+=1    
            #            self.progres_bar.after(500, self.Update_progres(a))
            #            self.run_frame.update()
            #            self.style_run.configure('text.Horizontal.TProgressbar',
            #                text='{:g} %'.format(((a/self.maximum)*100))) 
            #    self.total_def[:,:,2] = unpatchify(def_img_branch, (512*self.patches_image.shape[0],512*self.patches_image.shape[1]))
            #    if self.img_label_list.count('Total_img.png')!=1:
            #        self.img_label_list.append('Total_img.png')



            
        
        
        
        


        
        
        
        
            
            
            
            

        



    
        

            

            
        
    
                    


        






        return
    def Export_data(self): #Function to extract the data when pressed  the button extract data
        if self.patches_def is not None: #Aks for output path
            filepath =filedialog.askdirectory()
            if filepath:
                print(filepath)
                zf = zipfile.ZipFile(filepath+'/genrated_data.zip',
                    mode='w', 
                    compression=zipfile.ZIP_DEFLATED)   #generate empty zip file
                a=0
                
                
                
                
                if self.img_label_list:
                
                    different_diameter = []
                    different_unet_diameter =[]
                    structure_string = []
                    
                    
                    diameter_class = np.array(np.array(self.diameter_width_array)*self.pixel_width*self.digital_zoom,dtype = float)
                    diameter_class = np.round(diameter_class,1)
                    self.buf_file = io.BytesIO()
                    writer = pd.ExcelWriter(self.buf_file, engine='xlsxwriter') 
                    
                    print(len(diameter_class)) 


                    if np.all(self.patches_def[:,:,:,0] !=1): #Check if array is filled with output numbers
                        print('alg')
                        df1 = pd.DataFrame(columns = self.img_label_list)
                        f=0
                        for z in self.img_label_list:
                            data = []    
                            if  f == len(self.img_label_list)-1: #When last label is reached the total image has to be extracted
                                print('last')
                                self.fig_input,ax1 = plt.subplots() #Make pyplot figure
                                i = ax1.imshow(self.total_def[:,:,0]) #input image in figure
                                ax1.get_xaxis().set_visible(False) #Turn off x-as
                                divider = make_axes_locatable(ax1) #Make devider for better posistioning of colorbar
                                cax = divider.new_vertical(size = '5%', pad = 0.01, pack_start = True) #Make frame for colorbar
                                self.fig_input.add_axes(cax) # Add frame to figure
                                cbar =plt.colorbar(i,orientation='horizontal', cax= cax) #Add colorbar to figure
                                cbar.ax.set_xlabel('Diameter (\u03BCm)') #set title to colobar
                                self.buf = io.BytesIO() 
                                plt.savefig(self.buf) #Change pyplot structure to bytes
                                zf.writestr('Images/Algorithm/Algorithm'+str(z)+'', self.buf.getvalue()) #Save in zip file
                                plt.close()
                                numbers, amount, ignore,ignore = self.extract_data(None,self.total_def[:,:,0],None) #extract data from images
                                print(numbers)
                                for i in diameter_class: #Check if all classes are in numbers and make sure
                                    if different_diameter.count(str(i)+'\u03BCm') != 1:
                                        different_diameter.append(str(i)+'\u03BCm')
                                amount = amount.tolist()
                                for i in diameter_class: #Check if for all classes an amount is counted if not  a '-' will be added to keep every colummshape the same
                                    if i not in numbers:           
                                        miss_location = np.where(diameter_class==i)
                                        amount.insert(int(miss_location[0]),'-')
                                for i in amount:
                                    data.append(i) 
                                df1[str(z)] = data
                                f+=1
                                #This is done for every class including all the patches
                            else:
                                self.fig_input,ax1 = plt.subplots()
                                i = ax1.imshow(self.patches_def[f,:,:,0])
                                ax1.get_xaxis().set_visible(False)
                                divider = make_axes_locatable(ax1)
                                cax = divider.new_vertical(size = '5%', pad = 0.01, pack_start = True)
                                self.fig_input.add_axes(cax)
                                cbar =plt.colorbar(i,orientation='horizontal', cax= cax) 
                                cbar.ax.set_xlabel('Diameter (\u03BCm)')
                                self.buf = io.BytesIO()
                                plt.savefig(self.buf)
                                zf.writestr('Images/Algorithm/Algorithm_'+str(z)+'', self.buf.getvalue())
                                plt.close()
                                numbers, amount, ignore,ignore = self.extract_data(None,self.patches_def[f,:,:,0],None)
                                for i in diameter_class:
                                    if different_diameter.count(str(i)+'\u03BCm') != 1:
                                        different_diameter.append(str(i)+'\u03BCm')
                                amount = amount.tolist()
                                for i in diameter_class:
                                    if i not in numbers:        
                                        miss_location = np.where(diameter_class ==i)
                                        amount.insert(int(miss_location[0]),'-')
                                for i in amount:
                                    data.append(i)
                                print(len(amount))   
                                df1[str(z)] = data
                                f+=1 
                        df1= df1.assign(**{'Data Overview' : np.array(different_diameter)}) #assign in datafram
                        df1 = df1.set_index('Data Overview') #Set the right column as index 
                        df1.to_excel(writer, sheet_name = "Algorithm",index =True, header=True)

                        
                        

                    if np.all(self.patches_def[:,:,:,1] !=1):
                        df2 = pd.DataFrame(columns = self.img_label_list) #making data frame with the patch names as colums
                        f= 0
                        for z in self.img_label_list:
                            data = []    
                            print('Unet')
                            if  f == len(self.img_label_list)-1:
                                self.fig_input,ax1 = plt.subplots()
                                i = ax1.imshow(self.total_def[:,:,1])
                                ax1.get_xaxis().set_visible(False)
                                divider = make_axes_locatable(ax1)
                                cax = divider.new_vertical(size = '5%', pad = 0.01, pack_start = True)
                                self.fig_input.add_axes(cax)
                                cbar =plt.colorbar(i,orientation='horizontal', cax= cax)    
                                cbar.ax.set_xlabel('Diameter (\u03BCm)')
                                self.buf = io.BytesIO()
                                plt.savefig(self.buf)
                                zf.writestr('Images/Unet/Prediction'+str(z)+'', self.buf.getvalue())
                                plt.close()
                                numbers, amount, ignore,ignore = self.extract_data(None,self.total_def[:,:,1],None)
                                for i in diameter_class:
                                    if different_unet_diameter.count(str(i) +'\u03BCm') != 1:
                                        different_unet_diameter.append(str(i)+'\u03BCm')
                                amount = amount.tolist()
                                for i in diameter_class:
                                    if i not in numbers:        
                                        miss_location = np.where(diameter_class ==i)
                                        amount.insert(int(miss_location[0]),'-')
                                for i in amount:
                                    data.append(i)
                                df2[str(z)] = data
                                f+=1   
                            else:
                                self.fig_input,ax1 = plt.subplots()
                                i = ax1.imshow(self.patches_def[f,:,:,1])
                                ax1.get_xaxis().set_visible(False)
                                divider = make_axes_locatable(ax1)
                                cax = divider.new_vertical(size = '5%', pad = 0.01, pack_start = True)
                                self.fig_input.add_axes(cax)
                                cbar =plt.colorbar(i,orientation='horizontal', cax= cax) 
                                cbar.ax.set_xlabel('Diameter (\u03BCm)')
                                self.buf = io.BytesIO()
                                plt.savefig(self.buf)
                                zf.writestr('Images/Unet/Prediction_'+str(z)+'', self.buf.getvalue())
                                plt.close()
                                numbers, amount, ignore,ignore = self.extract_data(None,self.patches_def[f,:,:,1],None)
                                for i in diameter_class:
                                    if different_unet_diameter.count(str(i) +'\u03BCm') != 1:
                                        different_unet_diameter.append(str(i)+'\u03BCm')
                                amount = amount.tolist()
                                for i in diameter_class:
                                    if i not in numbers:        
                                        miss_location = np.where(diameter_class ==i)
                                        amount.insert(int(miss_location[0]),'-')
                                for i in amount:
                                    data.append(i)
                                df2[str(z)] = data
                                f+=1   
                        
                        df2= df2.assign(**{'Data Overview' : np.array(different_unet_diameter)}) #assign in dataf
                        df2 = df2.set_index('Data Overview') #Set the right column as index
                        df2.to_excel(writer, sheet_name = "Unet",index =True, header=True)
                                
                        #if np.all(self.patches_def[:,:,:,2] !=1):
                        #    
                        #    print('branch')
                        #    if  f == len(self.img_label_list)-1:
                        #        self.fig_input,ax1 = plt.subplots()
                        #        i = ax1.imshow(self.total_def[:,:,2])
                        #        img_split = np.where(image==5)
                        #        for k in range(0,len(img_split[0])):
                        #            y1,x1 = img_split[0][k], img_split[1][k]
                        #            ax1.plot(x1,y1,'bo')
                        #    
                        #        ax1.get_xaxis().set_visible(False)
                        #        divider = make_axes_locatable(ax1)
                        #        cax = divider.new_vertical(size = '5%', pad = 0.01, pack_start = True)
                        #        self.fig_input.add_axes(cax)
                        #        cbar =plt.colorbar(i,orientation='horizontal', cax= cax) 
                        #        cbar.ax.set_xlabel('Pixels')
                        #        self.buf = io.BytesIO()
                        #        plt.savefig(self.buf)
                        #        zf.writestr('Images/Branch/Branch'+str(z)+'', self.buf.getvalue())
                        #        plt.close()
                        #        ignore,ignore, structure,amount = self.extract_data(None,self.total_def[:,:,2],None)
                        #        if structure_string.count(str(structure)) != 1:
                        #            structure_string.append(str(structure))
                        #        data.append(amount)
                        #    
                        #    
                        #    else:
                        #        if a == 0:
                        #            self.fig_input,ax1 = plt.subplots()
                        #            i = ax1.imshow(self.img_patch_normal[f,:,:])
                        #            self.buf = io.BytesIO()
                        #            plt.savefig(self.buf)
                        #            zf.writestr('Images/Raw/normal_'+str(z)+'', self.buf.getvalue())
                        #            plt.close()
                        #            a=1
                        #        self.fig_input,ax1 = plt.subplots()
                        #        image = ax1.imshow(self.patches_def[f,:,:,2])
                        #        img_split = np.where(image==5)
                        #        for k in range(0,len(img_split[0])):
                        #            y1,x1 = img_split[0][k], img_split[1][k]
                        #            ax1.plot(x1,y1,'bo')
                        #        self.buf = io.BytesIO()
                        #        plt.savefig(self.buf)
                        #        zf.writestr('Images/Branch/Branch_'+str(z)+'', self.buf.getvalue())
                        #        plt.close()
                        #        ignore,ignore, structure,amount = self.extract_data(None,self.patches_def[f,:,:,2],None)

                        #        if structure_string.count(str(structure)) != 1:
                        #            structure_string.append(str(structure)
                        #        data.append(amount)
   
                            
                        
                         #Every column will be linked to the right extracted data
                        #f += 1           
                    f=0                
                    for z in self.img_label_list:
                        self.fig_input,ax1 = plt.subplots()
                        if  f == len(self.img_label_list)-1:
                            i = ax1.imshow(self.image)
                            self.buf = io.BytesIO()
                            plt.savefig(self.buf)
                            zf.writestr('Images/Raw/normal_'+str(z)+'', self.buf.getvalue())
                            plt.close()
                        else:
                            i = ax1.imshow(self.img_patch_normal[f,:,:])
                            self.buf = io.BytesIO()
                            plt.savefig(self.buf)
                            zf.writestr('Images/Raw/normal_'+str(z)+'', self.buf.getvalue())
                            f+=1
                            plt.close()    
                        
                     #Making a array of all the items inside the index list
                    
                    
                    

                    
                    writer.save()
                    zf.writestr('Data_sheet.xlsx', self.buf_file.getvalue())

                    


                    



                

                zf.close()
                print('made file')




        return  

if __name__ == "__main__":     
    window = tk.ThemedTk()
    window.get_themes()
    window.set_theme("radiance")
    window.geometry("1600x2500")
    window.config(bg='white')
    window.title("Diameter detection app")
    GUI(window)
    window.mainloop()
    
    
    

    

   
   
   
   
   
   
   
   



    # Windows in app
    


