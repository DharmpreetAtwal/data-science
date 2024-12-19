# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 10:47:06 2024

@author: Dharm
"""

import pandas as pd
import numpy as np

import os 

# %%
lst=[]
# Must cd into brain-tumor-mri-dataset
for (root,dirs,files) in os.walk("./data"):
    for file in files: 
        lst.append(os.path.join(root, file))
        
# %%
import matplotlib.pyplot as plt
import seaborn as sb
from glob import glob
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import keras
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Flatten, Dropout, Dense
from keras.optimizers import Adamax
from keras.metrics import Recall, Precision
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input

import matplotlib as mplt

# %%
tf.test.gpu_device_name()

data_path = "./data"
tumor_classes, paths = zip(*[(lbl, os.path.join(data_path, lbl, img))
                             for lbl in os.listdir(data_path) 
                                 if os.path.isdir(os.path.join(data_path, lbl))
                             for img in os.listdir(os.path.join(data_path, lbl))])

df = pd.DataFrame({'Tumor': tumor_classes, 'Path': paths})

# %%
# Visualize frequency of each class
def count_plot(df):
    plt.figure(figsize=(10,5))
    axis = sb.countplot(data=df, y=df['Tumor'])
    axis.bar_label(axis.containers[0])
    
count_plot(df)
    
# %%
df_train, df_test_val = train_test_split(df, train_size=0.8, 
                                         random_state=42, 
                                         stratify=df['Tumor'])
df_test, df_val = train_test_split(df_test_val, train_size=0.5)

# %%
# =============================================================================
# PREPROCESSING CHANGED
# MAY CAUSE ISSUES
# =============================================================================

count_plot(df_train)
count_plot(df_test)
count_plot(df_val)

# %%
# Preprocessing data into generators
batch = 32
size = (300, 300)    

idg_test = ImageDataGenerator(rescale=1/255)
idg = ImageDataGenerator(rescale=1/255, brightness_range=(0.8, 1.2))

gen_train = idg.flow_from_dataframe(df_train, 
                                    x_col='Path', y_col='Tumor',
                                    batch_size=batch, target_size=size)

gen_val = idg.flow_from_dataframe(df_val, 
                                  x_col='Path', y_col='Tumor',
                                  batch_size=batch, target_size=size)

gen_test = idg.flow_from_dataframe(df_test, 
                                   x_col='Path', y_col='Tumor',
                                   batch_size=batch, target_size=size)

# %%









    
    



