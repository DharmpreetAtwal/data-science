# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 10:47:06 2024

@author: Dharm
"""

import pandas as pd
import numpy as np
import os 

import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dropout, Dense, Rescaling, RandomBrightness
from keras.optimizers import Adamax
from keras.metrics import Recall, Precision

# %%
lst=[]
# Must cd into brain-tumor-mri-dataset
for (root,dirs,files) in os.walk("./data"):
    for file in files: 
        lst.append(os.path.join(root, file))
       
# %%
# Test GPU connected
tf.test.gpu_device_name()

# Creating df for Tumor and Path
data_path = "./data"

lst_images = []
# For each file in ./data
for tumor_type in os.listdir(data_path):
    if tumor_type == "train_test_val":
        continue
    
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
df_test, df_val = train_test_split(df_test_val, train_size=0.5, 
                                   stratify=df_test_val['Tumor'])
df_train.reset_index()
df_test.reset_index()
df_val.reset_index()

# %%
# Create seperate folders for train, test, val
ttv_path = "./data/train_test_val"
if not os.path.isdir(ttv_path):
    os.mkdir(ttv_path)


import shutil

train_val_path = "./data/train_test_val/train_val"
if not os.path.isdir(train_val_path):
    os.mkdir(train_val_path)
    
    for tumor in df['Tumor'].unique():
        os.mkdir(train_val_path + "/" + tumor)
        
    for i, row in df_train.iterrows():
        shutil.copy(row['Path'], train_val_path + "/" + row['Tumor'])
    for i, row in df_val.iterrows():
        shutil.copy(row['Path'], train_val_path + "/" + row['Tumor'])

test_path = "./data/train_test_val/test"
if not os.path.isdir(test_path):
    os.mkdir(test_path)
    
    for tumor in df['Tumor'].unique():
        os.mkdir(test_path + '/' + tumor)
    
    for i, row in df_test.iterrows():
        shutil.copy(row['Path'], test_path + '/' + row['Tumor'])

    
# %%
count_plot(df_train)
count_plot(df_test)
count_plot(df_val)

# %%
batch = 16
size = (299, 299)   

train_ds = tf.keras.utils.image_dataset_from_directory(
  train_val_path,
  validation_split=1/9,
  label_mode='categorical',
  subset="training",
  seed=123,
  image_size=size,
  batch_size=batch)

val_ds = tf.keras.utils.image_dataset_from_directory(
  train_val_path,
  validation_split=1/9,
  label_mode='categorical',
  subset="validation",
  seed=123,
  image_size=size,
  batch_size=batch)

test_ds = tf.keras.utils.image_dataset_from_directory(
  test_path,
  label_mode='categorical',
  seed=123,
  image_size=size,
  batch_size=batch,
  shuffle=False)

class_lst = train_ds.class_names

# %%
plt.figure(figsize=(18, 18))
dim = 4

for imgs, lbls in val_ds.take(1):
    for i in range(20):
        if i + 1 == dim ** 2 + 1:
            break
        
        axis = plt.subplot(dim, dim, i + 1)
        plt.imshow(imgs[i].numpy().astype("uint8"))
        
        
        plt.title(class_lst[np.argmax(lbls[i])])
        plt.axis("off")
    
plt.show()

# %%
auto_tune = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=auto_tune)
val_ds = val_ds.cache().prefetch(buffer_size=auto_tune)

input_shape = (299, 299, 3)
xception = tf.keras.applications.Xception(input_shape=input_shape,                                       
                                       weights="imagenet",
                                       include_top=False,
                                       pooling="max")

clf = Sequential()

clf.add(RandomBrightness(factor=0.2))
clf.add(Rescaling(1./255))
clf.add(xception)
clf.add(Flatten())
clf.add(Dropout(0.3))
clf.add(Dense(128, activation='relu'))
clf.add(Dropout(0.25))
clf.add(Dense(4, activation="softmax"))

clf.compile(Adamax(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', Recall(), Precision()])

# %%
# Originally used 10 epochs, but led to overfitting
# Best acc, pre, recall came with 7 epochs
with tf.device(tf.test.gpu_device_name()):
    h = clf.fit(train_ds, epochs=7, 
                   validation_data=val_ds,
                   shuffle=False)

# %%
# Mapping model performance over epochs
acc_train = h.history['accuracy']
loss_train = h.history['loss']
pre_train = h.history['precision_8']
recall_train = h.history['recall_8']

acc_val = h.history['val_accuracy']
loss_val = h.history['val_loss']
pre_val = h.history['val_precision_8']
recall_val = h.history['val_recall_8']

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
score_train = clf.evaluate(train_ds, verbose=1)
score_valid = clf.evaluate(test_ds, verbose=1)
score_test = clf.evaluate(val_ds, verbose=1)

# %%
pred = clf.predict(test_ds)
y_pred = np.argmax(pred, axis=1)

# %%
y_true = []
for _, labels in test_ds:
    for item in labels.numpy():    
        y_true.append(np.argmax(item))
    
# %%
matrix = confusion_matrix(y_true, y_pred)
lbls = list(test_ds.class_names)

plt.figure(figsize=(10, 8))

sb.heatmap(matrix, cmap='mako',annot=True, fmt='d', xticklabels=lbls, yticklabels=lbls)
plt.show()




    
    



