import pandas as pd
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV

from imblearn.under_sampling import RandomUnderSampler
# %%
# Don't include NaN in list of na_values, considered a valid value
df = pd.read_csv('./diabetes2/diabetes+130-us+hospitals+for+years+1999-2008/diabetic_data.csv', 
                 na_values=['?'], keep_default_na=False)

# Columns that are missing 50% of their data
half_empty_columns = df.columns[(df.isna() | df.eq("")).sum() / len(df) > 0.5].tolist()

# %%
# Drop columns that are indep. of discharge decision, or half empty
df.drop(['encounter_id', 'patient_nbr'] + half_empty_columns, axis=1, inplace=True)
df_clean = df.dropna()

# %%
# Binarize the readmitted label we're trying to predict
# <30 => 'readmitted' = 1
# Any other category => 'readmitted' = 0
df_clean.loc[df_clean['readmitted'] == "<30", 'readmitted'] = 1
df_clean.loc[df_clean['readmitted'] != 1, 'readmitted'] = 0
df_clean['readmitted'] = df_clean['readmitted'].astype(int) 


# =============================================================================
# TESTING WITH 5000 SAMPLE
# =============================================================================
#df_clean = df_clean.sample(5000, random_state=42)



# Separate into features and labels
df_features = df_clean.drop(['readmitted'], inplace=False, axis=1).copy()
df_labels = df_clean['readmitted'].copy()
df_labels = df_labels.reset_index(drop=True)

# %%
# Seperate Categorical Features into High/Low Cardinalities
low_cardinality_cat = ['race', 'gender', 'age', 'admission_type_id', 
                       'discharge_disposition_id', 'admission_source_id', 
                       'payer_code', 'medical_specialty', 'max_glu_serum', 
                       'A1Cresult', 'metformin', 'repaglinide', 'nateglinide', 
                       'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 
                       'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 
                       'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide', 
                       'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 
                       'glimepiride-pioglitazone', 'metformin-rosiglitazone', 
                       'metformin-pioglitazone', 'change', 'diabetesMed']
high_cardinality_cat = ['diag_1', 'diag_2', 'diag_3']

# Apply One-Hot Encoding to Low Cardinality Categorical Features
df_features = pd.get_dummies(df_features, columns=low_cardinality_cat)

# Apply Frequency Encoding to High Cardinality Categorical Features
for feature in high_cardinality_cat:
    freq_encoding = df_features[feature].value_counts(normalize=True)
    df_features[feature + "_freq"] = df_features[feature].map(freq_encoding)
df_features.drop(high_cardinality_cat, inplace=True, axis=1)

# %%
# Use RandomUnderSampler to create an artificial dataset with no class imbalance
rus = RandomUnderSampler(random_state=42)
X, y = rus.fit_resample(df_features, df_labels)

# %%
# Perform Random Forest feature selection on this artificial dataset 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
matrix = confusion_matrix(y_test, y_pred)

# Display results of Random Forest
display = ConfusionMatrixDisplay(matrix)
display.plot()
plt.show()

# %%
# Display 20 most important features
f_i = list(zip(X_train.columns, rf.feature_importances_))
f_i.sort(key=lambda x: x[1], reverse=True)

most = f_i[0:20]
plt.barh([x[0] for x in most], [x[1] for x in most])

# %%
# Recursively eliminate least important features using cross validation
rfecv = RFECV(rf, cv=5, scoring='recall')
rfecv.fit(X_train, y_train)
selected = X_train.columns[rfecv.support_]

# %%
# After appyling reduction, see improvement in artificial dataset
X_train, X_test, y_train, y_test = train_test_split(X[selected], y, test_size=0.5, random_state=42)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
matrix = confusion_matrix(y_test, y_pred)

# Display results of Random Forest
display = ConfusionMatrixDisplay(matrix)
display.plot()
plt.show()

# %%
param_grid = {
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# Exhaustive search for best hyperparameters that maximize recall
grid_search = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', 
                           random_state=42),
    param_grid,
    scoring='recall',
    cv=5
)

# Use reduced feature set
df_features_reduced = df_features[selected]

grid_search.fit(df_features_reduced, df_labels)
best_model = grid_search.best_estimator_

# %%
# Train test optimized RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(
    df_features_reduced, df_labels, test_size=0.5, random_state=42)
best_model.fit(X_train, y_train)

#y_pred = best_model.predict(X_test)

# Adjust threshold to trade higher recall for lower precision
y_prob = best_model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.40).astype(int)

matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
disp.plot()
plt.show()

# %%

# %%
m = ['o', 'x']
c = ['hotpink', 'red']

fig = plt.figure()
ax = fig.add_subplot(111)

for i in range(len(pca_components)):
    if df_labels.at[i] == 0:
        marker_index = 0
    else:
        marker_index = 1

    if marker_index != 0:
        continue

    ax.scatter(pca_components[i][0], pca_components[i][1],
                color=c[pred_labels[i]],
                marker=m[marker_index])


