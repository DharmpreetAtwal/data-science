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
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.preprocessing import PCA

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
df_clean.drop(['date_time', 'srch_ci', 'srch_co'], inplace=True, axis=1)

#df_clean = df_clean.sample(5000)
# %%
# =============================================================================
# # POTENTIAL ISSUE: May need to encode ID based features that aren't randomly assigned
# posa_continent is directly related to site_name
# =============================================================================

# Drop label, and randomly generated id's 
df_features = df_clean.drop(['is_booking', 'Unnamed: 0', 'user_id'], inplace=False, axis=1)

# Apply One Hot Encoding to low cardinality categorical data
low_cardinal_cat = ['site_name', 'posa_continent', 'user_location_country', 'channel', 'srch_destination_type_id', 'hotel_continent']
df_features = pd.get_dummies(df_features, columns=low_cardinal_cat)

# Apply Frequency Encoding to high cardinality categorical data
# Replace high cardinality features with their frequency encoded versions
high_cardinal_cat = ['user_location_region', 'user_location_city', 'srch_destination_id', 'hotel_country', 'hotel_market', 'hotel_cluster']
for feature in high_cardinal_cat:
    freq_encoding = df_features[feature].value_counts(normalize=True)
    df_features[feature + "_Encoded"] = df_features[feature].map(freq_encoding)
df_features.drop(high_cardinal_cat, inplace=True, axis=1)

df_labels = df_clean['is_booking'].reset_index(drop=True)

# %%
# Use RandomUnderSampler to create a artificial dataset with no class imbalance
rus = RandomUnderSampler(random_state=42)
X, y = rus.fit_resample(df_features, df_labels)

# Perform feature selection on this artificial dataset using Random Forest
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
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
rfecv = RFECV(rf, cv=5, scoring='recall')
rfecv.fit(X_train, y_train)
selected = X_train.columns[rfecv.support_]

# %%
param_grid = {
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# Exhaustive search for best hyperparameters that maximize recall
grid_search = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42),
    param_grid,
    scoring='recall',
    cv=5
)

df_features_reduced = df_features[selected]
grid_search.fit(df_features_reduced, df_labels)
best_model = grid_search.best_estimator_

# %%
# Train test optimized RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(df_features_reduced, df_labels, test_size=0.5, random_state=42)
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)
matrix = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
disp.plot()
plt.show()

# %%
df_clean = df_clean.sample(5000)



kmeans = KMeans(n_clusters=2, random_state=42)
pred_labels = kmeans.fit_predict(df_features_reduced, df_labels)

pca = PCA(n_components=2)
pca_components = pca.fit_transform(df_features_reduced)

m = ['o', 'x']
c = ['hotpink', 'red']

fig = plt.figure()
ax = fig.add_subplot(111)

for i in range(len(pca_components)):
    if df_labels.at[i] == 0:
        marker_index = 0
    else:
        marker_index = 1

    #if marker_index != 0:
    #    continue

    ax.scatter(pca_components[i][0], pca_components[i][1],
                color=c[pred_labels[i]],
                marker=m[marker_index])



        







