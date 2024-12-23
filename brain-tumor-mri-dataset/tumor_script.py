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

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dropout, Dense
from keras.optimizers import Adamax
from keras.metrics import Recall, Precision
from keras.preprocessing.image import ImageDataGenerator

# %%
# Test GPU connected
tf.test.gpu_device_name()

# Creating df for Tumor and Path
data_path = "./data"

lst_images = []
# For each file in ./data
for tumor_type in os.listdir(data_path):
    
    # Check that ./data_path/tumor_type is a folder
    if os.path.isdir(os.path.join(data_path, tumor_type)):
        
        # For each image in the folder ./data_path/tumor_type
        for image in os.listdir(os.path.join(data_path, tumor_type)):
            
            # Append the tuple: (tumor_type, ./data_path/tumor_type/image)
            lst_images.append((tumor_type, os.path.join(data_path, tumor_type, image)))

tumor_classes, paths = zip(*lst_images)
df = pd.DataFrame({'Tumor': tumor_classes, 'Path': paths})

# %%
# Visualize frequency of each class
def count_plot(df):
    plt.figure(figsize=(10,5))
    axis = sb.countplot(data=df, y=df['Tumor'])
    axis.bar_label(axis.containers[0])
    
count_plot(df)
    
# %%
# Split into train and test/val
df_train, df_test_val = train_test_split(df, train_size=0.8,                              
                                         stratify=df['Tumor'],
                                         random_state=42)
df_test, df_val = train_test_split(df_test_val, train_size=0.5)

# %%
count_plot(df_train)
count_plot(df_test)
count_plot(df_val)

# %%
# Preprocessing data into generators
# Large Batch size causes ResourceExhaustionError
batch = 16
size = (299, 299)    

idg = ImageDataGenerator(rescale=1/255, brightness_range=(0.8, 1.2))
idg_test = ImageDataGenerator(rescale=1/255)

gen_train = idg.flow_from_dataframe(df_train, 
                                    x_col='Path', y_col='Tumor',
                                    batch_size=batch, target_size=size)

gen_val = idg.flow_from_dataframe(df_val, 
                                  x_col='Path', y_col='Tumor',
                                  batch_size=batch, target_size=size)

gen_test = idg_test.flow_from_dataframe(df_test, 
                                   x_col='Path', y_col='Tumor',
                                   batch_size=batch, target_size=size, 
                                   shuffle=False)

# %%
class_lst = list(gen_train.class_indices.keys())
dict_class = gen_train.class_indices
imgs, lbls = next(gen_test)

plt.figure(figsize=(18, 18))

dim = 4
for i, (img, lbl) in enumerate(zip(imgs, lbls)):
    if i + 1 == dim ** 2 + 1:
        break
    
    class_name = class_lst[np.argmax(lbl)]
    
    plt.title(class_name, fontsize=16)
    plt.subplot(dim, dim, i + 1)
    plt.imshow(img)
    
plt.show()

# %%
input_shape = (299, 299, 3)
xception = tf.keras.applications.Xception(input_shape=input_shape,                                       
                                       weights="imagenet",
                                       include_top=False,
                                       pooling="max")

clf = Sequential()

clf.add(xception)
clf.add(Flatten())
clf.add(Dropout(0.3))
clf.add(Dense(128, activation='relu'))
clf.add(Dropout(0.25))
clf.add(Dense(4, activation="softmax"))

clf.compile(Adamax(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', Recall(), Precision()])
clf.summary()

# %%
with tf.device(tf.test.gpu_device_name()):
    history = clf.fit(gen_train, epochs=10, 
                   validation_data=gen_val,
                   shuffle=False)

# %%
# Mapping model performance over epochs
h = history
acc_train = h.history['accuracy']
loss_train = h.history['loss']
pre_train = h.history['precision']
recall_train = h.history['recall']

acc_val = h.history['val_accuracy']
loss_val = h.history['val_loss']
pre_val = h.history['val_precision']
recall_val = h.history['val_recall']

acc_index = np.argmax(acc_val)
highest_acc = acc_val[acc_index]

loss_index = np.argmin(loss_val)
lowest_val = loss_val[loss_index]

pre_index = np.argmax(pre_val)
highest_pre = pre_val[pre_index]

recall_index = np.argmax(recall_val)
highest_recall = recall_val[recall_index]

epochs = range(1, len(acc_train) + 1)

lbl_acc = f'Best = {str(acc_index + 1)}'
lbl_loss = f'Best = {str(loss_index + 1)}'
lbl_pre = f'Best = {str(acc_index + 1)}'
lbl_recall = f'Best = {str(recall_index + 1)}'

plt.figure(figsize=(18, 15))

plt.subplot(2, 2, 1)
plt.plot(epochs, loss_train, 'g', label='Train Loss')
plt.plot(epochs, loss_val, 'b', label='Val Loss')
plt.scatter(loss_index + 1, lowest_val, s=100, c='red', label=lbl_loss)
plt.title("Train and Val Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(epochs, acc_train, 'g', label='Train Acc')
plt.plot(epochs, acc_val, 'b', label='Val Acc')
plt.scatter(acc_index + 1, highest_acc, s=100, c='red', label=lbl_acc)
plt.title("Train and Val Acc")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(epochs, pre_train, 'g', label='Train Pre')
plt.plot(epochs, pre_val, 'b', label='Val Pre')
plt.scatter(pre_index + 1, highest_pre, s=100, c='red', label=lbl_pre)
plt.title("Train and Val Pre")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(epochs, recall_train, 'g', label='Train Recall')
plt.plot(epochs, recall_val, 'b', label='Val Recall')
plt.scatter(recall_index + 1, highest_recall, s=100, c='red', label=lbl_pre)
plt.title("Train and Val Recall")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# %%
#clf.save("./model.keras")
#clf = keras.models.load_model("model.keras")

# %%
gen_train_copy = gen_train
gen_test_copy = gen_test
gen_val_copy = gen_val

score_train = clf.evaluate(gen_train_copy, verbose=1)
# %%
score_valid = clf.evaluate(gen_test_copy, verbose=1)
score_test = clf.evaluate(gen_val_copy, verbose=1)

# %%
pred = clf.predict(gen_test_copy)
y_pred = np.argmax(pred, axis=1)

# %%
matrix = confusion_matrix(gen_test_copy.classes, y_pred)
lbls = list(dict_class)

plt.figure(figsize=(10, 8))

sb.heatmap(matrix, cmap='mako',annot=True, fmt='d', xticklabels=lbls, yticklabels=lbls)
plt.show()




    
    



