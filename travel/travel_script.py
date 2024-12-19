# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 13:31:20 2024

@author: dharm
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from keras import Sequential
from keras.layers import Dense

from imblearn.under_sampling import RandomUnderSampler

# %%
df = pd.read_csv("./travel/travel.csv")

# Remove columns where 90% of the data is nan
nan_col = df.columns[(df.isna() | df.eq("")).sum() / len(df) > 0.9].tolist()

# Remove rows missing data
df_clean = df.dropna().copy()

# Convert date-time strings into day of year values ranging from 1-366
# =============================================================================
# # POTENTIAL ISSUE: Dates 2013/12/31 and 2014/01/01, would be considered "far apart"
# =============================================================================

df_clean['date_of_year'] = pd.to_datetime(df_clean['date_time']).dt.dayofyear.astype(int)
df_clean['check_in'] = pd.to_datetime(df_clean['srch_ci']).dt.dayofyear.astype(int)
df_clean['check_out'] = pd.to_datetime(df_clean['srch_co']).dt.dayofyear.astype(int)
df_clean.drop(['date_time', 'srch_ci', 'srch_co', 'Unnamed: 0', 'user_id'], inplace=True, axis=1)

# %%
# =============================================================================
# # POTENTIAL ISSUE: Need to encode ID based features that aren't randomly assigned
# posa_continent is directly related to site_name
# =============================================================================
# Apply One Hot Encoding to low cardinality categorical data
low_cardinal_cat = ['site_name', 'posa_continent', 'user_location_country', 'channel', 'srch_destination_type_id', 'hotel_continent']
df_clean = pd.get_dummies(df_clean, columns=low_cardinal_cat)

# Apply Frequency Encoding to high cardinality categorical data
# Replace high cardinality features with their frequency encoded versions
high_cardinal_cat = ['user_location_region', 'user_location_city', 'srch_destination_id', 'hotel_country', 'hotel_market', 'hotel_cluster']
for feature in high_cardinal_cat:
    freq_encoding = df_clean[feature].value_counts(normalize=True)
    df_clean[feature + "_Encoded"] = df_clean[feature].map(freq_encoding)
df_clean.drop(high_cardinal_cat, inplace=True, axis=1)

# Drop label, and randomly generated id's 
df_features = df_clean.drop(['is_booking'], inplace=False, axis=1)
df_labels = df_clean['is_booking'].reset_index(drop=True)

# %%
# Use RandomUnderSampler to create a artificial dataset with no class imbalance
rus = RandomUnderSampler(random_state=42)
X_balance, y_balance = rus.fit_resample(df_features, df_labels)

# Perform feature selection on this artificial dataset using Random Forest
X_train, X_test, y_train, y_test = train_test_split(X_balance, y_balance, test_size=0.5, random_state=42)
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
matrix = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
disp.plot()
plt.show()

# %%
# Display the 20 most important features
f_i = list(zip(X_train.columns, rf.feature_importances_))
f_i.sort(key=lambda x: x[1], reverse=True)

most = f_i[0:20]
plt.barh([x[0] for x in most], [x[1] for x in most])

# %%
# Recursively eliminate the least important features with cross validation
# rfecv = RFECV(rf, cv=5, scoring='recall')
# rfecv.fit(X_train, y_train)
# selected = X_train.columns[rfecv.support_]

# The result of runnning the above RFECV (long runtime)
selected = ['orig_destination_distance', 'is_mobile', 'is_package',
       'srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt', 'cnt',
       'date_of_year', 'check_in', 'check_out', 'site_name_2', 'site_name_8',
       'site_name_11', 'site_name_34', 'posa_continent_1', 'posa_continent_3',
       'user_location_country_66', 'user_location_country_205', 'channel_0',
       'channel_1', 'channel_2', 'channel_3', 'channel_4', 'channel_5',
       'channel_9', 'srch_destination_type_id_1', 'srch_destination_type_id_3',
       'srch_destination_type_id_4', 'srch_destination_type_id_5',
       'srch_destination_type_id_6', 'hotel_continent_2', 'hotel_continent_3',
       'hotel_continent_4', 'hotel_continent_6',
       'user_location_region_Encoded', 'user_location_city_Encoded',
       'srch_destination_id_Encoded', 'hotel_country_Encoded',
       'hotel_market_Encoded', 'hotel_cluster_Encoded']

# %%
# # Get the reduced feature subset of the sample
X = df_clean[selected].values
y = df_clean[['is_booking']].values

scaler=StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

neg, pos = np.bincount(df_clean['is_booking'])
weight_0 = (1 / neg) * (len(X) / 2.0)
weight_1 = (1 / pos) * (len(X) / 2.0)

class_weight = {0: weight_0, 1: weight_1}

# %%
from keras.layers import Dropout, BatchNormalization
clf = Sequential()

# 40 Feature input layer
clf.add(Dense(units=32, kernel_initializer='uniform', activation='relu', input_dim=40,))

# 1st Hidden Layer
clf.add(Dense(units=128, kernel_initializer='uniform', activation='relu'))

# Dropout to prevent overfitting
clf.add(BatchNormalization())  
clf.add(Dropout(0.5))

# 2nd Hidden Layer
clf.add(Dense(units=64, kernel_initializer='uniform', activation='relu'))

# Dropout to prevent overfitting
clf.add(BatchNormalization())  
clf.add(Dropout(0.25))

# Output layer, binary classification
clf.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

from keras.metrics import Recall

mets=[Recall(name='recall')]
clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=mets)
ann = clf.fit(X_train,y_train, batch_size=10, epochs=40, class_weight=class_weight, verbose=1)

# %%
y_prob = clf.predict(X_test, batch_size=10)
y_pred = (y_prob > 0.5).astype(int)

matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
disp.plot()
plt.show()
# %%
clf.save("./travel/booking_epoch40.keras")


