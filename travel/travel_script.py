# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 13:31:20 2024

@author: dharm
"""
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv("C:/Users/coolb/Downloads/Coding/data-science/travel/travel.csv")
# %%

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
df_clean.drop(['date_time', 'srch_ci', 'srch_co'], inplace=True, axis=1)

df_clean = df_clean.sample(10000)
# %%

# =============================================================================
# # POTENTIAL ISSUE: May need to encode ID based features that aren't randomly assigned
# =============================================================================


df_features = df_clean.drop(['is_booking'], inplace=False, axis=1)
df_labels = df_clean['is_booking'].reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(df_features, df_labels, test_size=0.5, random_state=42, stratify=df_labels)

# %%
# RECALL

rf = RandomForestClassifier(class_weight="balanced", random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
matrix = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
disp.plot()
plt.show()






        







