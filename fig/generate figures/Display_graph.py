from tkinter.constants import BOTTOM, E, LEFT, N, RIGHT, S, TOP, W
from tkinter.filedialog import askopenfile, askopenfilename
import tensorflow as tf
import tensorflow_addons as tfa
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from collections import Counter

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from core.diameter_shortest_distance import diameter_short
from patchify import patchify, unpatchify
from core.NN.Instance_semantic_segmentation_model import unet_multiclass
from core.branch_detection import branch
from tkinter import ttk
from ttkthemes import themed_tk as tk
import tkinter as tkin
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pandas import DataFrame
from tkinter import filedialog
import zipfile
import io
import seaborn as sns
import pandas as pd
from skimage.transform import resize
import numpy as np




class display_graph():
    def __init__(self):      
        self.graph_window = tk.ThemedTk()
        self.graph_window.get_themes()
        self.graph_window.set_theme("radiance")
        self.graph_window.geometry("800x600")
        self.graph_window.config(bg='white')
        self.graph_window.title('View Graph')
        
        self.style_frame = ttk.Style(self.graph_window)
        self.style_frame.configure("TFrame", background = 'white')
        
        self.style_text = ttk.Style(self.graph_window)
        self.style_text. configure("TLabel", font = ('calibri', 13,'bold'), forground ='black', background= 'white' )
        
        self.style_checkbutton = ttk.Style(self.graph_window)
        self.style_checkbutton.configure("TCheckbutton", background="wihte", forground = "white")

        self.selection_frame_graph = ttk.Frame(self.graph_window, width = 400, height =500 ,style= "TFrame")
        self.plotting_graph_frame = ttk.Frame(self.graph_window, width = 400, height = 500 ,style= "TFrame")
        self.closing_frame_graph = ttk.Frame(self.graph_window, width = 400, height = 20, style= "TFrame" )
        

        
        self.selection_frame_graph.grid(column=0, row =0, padx=5)
        self.plotting_graph_frame.grid(column=1, row =0)
        self.closing_frame_graph.grid(row =1, column = 1, columnspan=2)
        
        selection_frame_text =ttk.Label(self.selection_frame_graph,text = "Selected ROI", style="TLabel")
        plotting_graph_frame_text = ttk.Label(self.plotting_graph_frame, text ="Diameter Prediction Data", style ="TLabel")
        self.checkbox_graph_1 = ttk.Checkbutton(self.selection_frame_graph,variable = GUI.selected_graph, onvalue= 0, offvalue= -1, style = "TCheckbutton", command= GUI.show_variable_graph)
        self.checkbox_graph_2 = ttk.Checkbutton(self.selection_frame_graph,variable = GUI.selected_graph, onvalue= 1, offvalue= -1, style = "TCheckbutton", command= GUI.show_variable_graph)
        self.checkbox_graph_3 = ttk.Checkbutton(self.selection_frame_graph,variable = GUI.selected_graph, onvalue= 2, offvalue= -1, style = "TCheckbutton", command= GUI.show_variable_graph)
        self.checkbox_graph_4 = ttk.Checkbutton(self.selection_frame_graph,variable = GUI.selected_graph, onvalue= 3, offvalue= -1, style = "TCheckbutton", command= GUI.show_variable_graph)
        self.button_close_graph_frame = ttk.Button(self.closing_frame_graph,text='Done',width= 20, command=GUI.close_window)

        selection_frame_text.grid(row = 3, columnspan = 4)
        plotting_graph_frame_text.grid(row=2)



        self.button_close_graph_frame.grid()
        self.checkbox_graph_1.grid(row = 1, column =0)
        self.checkbox_graph_2.grid(row = 1, column =2)
        self.checkbox_graph_3.grid(row =2, column =0)
        self.checkbox_graph_4.grid(row =2, column = 2)
        GUI.scetch_selection(self.selection_frame_graph)
        GUI.Make_chart(self.plotting_graph_frame,None,(4,4))

        self.graph_window.mainloop()