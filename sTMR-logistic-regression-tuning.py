import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression

# Load sRMT dataset
proj_folder = r"E:\work_local_backup\neuroma_data_project\TMR-ML"
data_folder = os.path.join(proj_folder, "data")
data_file = os.path.join(data_folder, "sTMR.csv")
df_secondary = pd.read_csv(data_file)

fig_folder = os.path.join(proj_folder, "figures", "sTMR")
if not os.path.exists(fig_folder):
    os.makedirs(fig_folder)

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

# use cross-validation to tune the logistic regression model
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=321)
logistic_regression = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000000)

# Define the parameter grid
param_grid = {
    # 'C': np.logspace(-4, 4, 1000),  # Regularization strength
    'class_weight': [None, 'balanced'],  # Adjust for class imbalance
    'C': np.linspace(0, 3, 1000),  # Regularization strength
    'tol': [1e-3],  # Tolerance for stopping criteria
    'solver': ['liblinear']  # Solver (could also try 'saga')
}

# Perform the grid search
grid_search = GridSearchCV(logistic_regression, param_grid, cv=cv, n_jobs=-1, scoring='roc_auc')
grid_search.fit(X_encoded, y)

# Get the best model
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"Best model: {best_model}")
print(f"Best params: {best_params}")
print(f"Best score: {best_score}")

'''
Best model: LogisticRegression(C=0.34234234234234234, class_weight='balanced',
                   max_iter=1000000, penalty='l1', solver='liblinear',
                   tol=0.001)
Best params: {'C': 0.34234234234234234, 'class_weight': 'balanced', 'solver': 'liblinear', 'tol': 0.001}
Best score: 0.818452380952381

'''
