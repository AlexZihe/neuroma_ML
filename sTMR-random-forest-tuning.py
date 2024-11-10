import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

# Load sRMT dataset
proj_folder = r"E:\work_local_backup\neuroma_data_project\TMR-ML"
data_folder = os.path.join(proj_folder, "data")
data_file = os.path.join(data_folder, "sTMR.csv")
df_secondary = pd.read_csv(data_file)

df_secondary = df_secondary.dropna()

# Define the target variable (dependent variable) as y
X = df_secondary.drop(columns=['good_outcome'])
y = df_secondary['good_outcome']

# Separate numerical and categorical columns
numerical_cols = X.select_dtypes(include='number').columns
categorical_cols = X.select_dtypes(include='object').columns

# One-hot encode the categorical columns
onehot_encoder = OneHotEncoder(drop='first', sparse=False)
onehot_encoded = onehot_encoder.fit_transform(X[categorical_cols])

# Standardize the numerical columns
scaler = StandardScaler()
scaled = scaler.fit_transform(X[numerical_cols])

# Combine the processed features
X_encoded = np.concatenate([scaled, onehot_encoded], axis=1)

# Generate the column names for the encoded features
encoded_columns = list(X[numerical_cols].columns)
for i, cat in enumerate(categorical_cols):
    encoded_columns.extend([f"{cat}_{category}" for category in onehot_encoder.categories_[i][1:]])

X_encoded = pd.DataFrame(X_encoded, columns=encoded_columns)

# Set up the RandomForest model
random_forest = RandomForestClassifier(random_state=321)

# Define the parameter grid for tuning
param_grid = {
    'n_estimators': [400,500,600,700],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 4, 10],  # Minimum number of samples required to be at a leaf node
    'max_features': [None, 'sqrt', 'log2'],  # Number of features to consider when looking for the best split
    'class_weight': [None, 'balanced'],  # Adjust for class imbalance
}

# Cross-validation setup
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=321)

# Perform the grid search
grid_search = GridSearchCV(random_forest, param_grid, cv=cv, n_jobs=-1, scoring='roc_auc')

grid_search.fit(X_encoded, y)

# Get the best model
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"Best model: {best_model}")
print(f"Best params: {best_params}")
print(f"Best score: {best_score}")


'''
Best model: RandomForestClassifier(class_weight='balanced', min_samples_leaf=10,
                       n_estimators=700, random_state=321)
Best params: {'class_weight': 'balanced', 'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 10, 'min_samples_split': 2, 'n_estimators': 700}
Best score: 0.8160714285714287

'''