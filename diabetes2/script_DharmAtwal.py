import pandas as pd
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# %%
df = pd.read_csv('./diabetes+130-us+hospitals+for+years+1999-2008/diabetic_data.csv', na_values=['?'], keep_default_na=True)
# %%

# Drop columns unrelated to discharge decision, or contain very little data
nan_columns = df.columns[(df.isna() | df.eq("")).sum() / len(df) > 0.1].tolist()
df.drop(['encounter_id', 'patient_nbr'] + nan_columns, axis=1, inplace=True)
df_clean = df.dropna()

# Binarize the readmitted label we're trying to predict
df_clean.loc[df_clean['readmitted'] == "<30", 'readmitted'] = 1
df_clean.loc[df_clean['readmitted'] != 1, 'readmitted'] = 0
df_clean['readmitted'] = df_clean['readmitted'].astype(int) 


## TESTING WITH SAMPLING
df_clean = df_clean.sample(5000, random_state=42)
# %%

# Separate into features and True labels
df_features = df_clean.drop(['readmitted'], inplace=False, axis=1)
df_labels = df_clean['readmitted']
df_labels = df_labels.reset_index(drop=True)

# Apply One-Hot Encoding 
objects = df_features.select_dtypes(include='object').columns.tolist()
df_features_encoded = pd.get_dummies(df_features, columns=objects)


# %%
X_train, X_test, y_train, y_test = train_test_split(df_features_encoded, df_labels, test_size=0.3, random_state=42, stratify=df_labels)
model = RandomForestClassifier(random_state=42)

# Backwards Feature Selection
# remain_df represents the set of features that remain, initially contains all
# Perform One-Hot-Encoding on remain_df only AFTER backwards selection is complete
remain_df = df_features.copy()

# Stopping condition is 20 features left
while len(remain_df.columns) > 20:
    # Refresh list of scores when deciding which feature to drop
    scores = []
    
    # Every iteration, we remove a feature and perform One-Hot-Encoding on what remains
    for feature in remain_df.columns:
        print(feature, len(remain_df.columns))
        
        # Create a temp df, remove 1 feature to test reduction
        reduced_temp = remain_df.copy()
        reduced_temp.drop([feature], inplace = True, axis=1)

        # Apply One Hot Encoding to features/label of type 'object'
        objects = reduced_temp.select_dtypes(include='object').columns.tolist()
        reduced_temp_encoded = pd.get_dummies(reduced_temp, columns=objects)

        # Train, test based on the reduced subset of features
        X_train_temp = X_train[reduced_temp_encoded.columns]
        X_test_temp = X_test[reduced_temp_encoded.columns]

        model.fit(X_train_temp, y_train)
        y_pred = model.predict(X_test_temp)

        score = accuracy_score(y_test, y_pred)
        scores.append((score, feature))

    print("COMPLETE")
        
    # Sort the Scores from highest to lowest accurary score
    scores.sort(reverse=True, key=lambda x: x[0])
    print("scores: ", scores)

    # Get the best feature to remove, lowest score at the end
    best_remove = scores[-1][1]
        
    print("removing: ", best_remove, scores[-1][0])
    remain_df.drop([best_remove], inplace = True, axis=1)

        


# %%
# remain_df after running backwards selection
remain_df = df_features.copy()
remain_df = remain_df[['gender', 'age', 'num_procedures', 'number_inpatient',
        'number_diagnoses', 'metformin', 'repaglinide', 'nateglinide',
        'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide',
        'miglitol', 'troglitazone', 'examide', 'citoglipton', 'insulin',
        'glipizide-metformin', 'glimepiride-pioglitazone', 'change']]

# # Perform One-Hot-Encoding on the remaining features
objects = remain_df.select_dtypes(include='object').columns.tolist()
df_features_encoded = pd.get_dummies(remain_df, columns=objects)

# # Scale data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df_features_encoded)

# %%
# Perform KMeans clustering
kmeans_cluster = KMeans(n_clusters=2, random_state=42)
labels = kmeans_cluster.fit_predict(data_scaled)

# Decompose into lower dimensional space
pca = PCA(n_components=2)
pca_components = pca.fit_transform(data_scaled)

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

    #if marker_index != 0:
    #    continue

    ax.scatter(pca_components[i][0], pca_components[i][1],
                color=c[labels[i]],
                marker=m[marker_index])


plt.xlabel('pca_1')
plt.ylabel('pca_2')
plt.title("Patient Re-admittance Clustering")
plt.show()
