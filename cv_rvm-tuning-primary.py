import numpy as np
from sklearn_rvm import EMRVC
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV

# Load data
# proj_folder = "/Users/zihealexzhang/work_local/neuroma_data_project/aim_1"
proj_folder = r"E:\work_local_backup\neuroma_data_project\aim_1"
data_folder = os.path.join(proj_folder, "data")
data_file = os.path.join(data_folder, "TMR_dataset_ML_March24.xlsx")

fig_folder = os.path.join(proj_folder, "figures", "primary_TMR")
df = pd.read_excel(data_file)

df_primary = df[df['timing_tmr'] == 'Primary']
df_primary = df_primary.drop(columns=['record_id',
                                      'participant_id',
                                      # 'mrn',
                                      'birth_date',
                                      'race',
                                      'adi_natrank',
                                      'adi_statrank',
                                      'employment_status',
                                      'insurance',
                                      'date_amputation',
                                      'time_preopscoretotmr',
                                      'date_injury_amputation',
                                      'type_surg_tmr',
                                      'time_amptmr_days',
                                      'date_surgery_ican',
                                      'date_discharge',
                                      'follow_up_years',
                                      'last_score',
                                      'type_surg_rpni',
                                      'mech_injury_amputation',
                                      'malignacy_dichotomous',
                                      'trauma_dichotomous',
                                      'timing_tmr',
                                      'time_amptmr_years',
                                      'age_ican_surgery',
                                      'pain_score_difference',
                                      'MCID',
                                      'preop_score',
                                      'pain_mild',
                                      'pain_disappearance',
                                      'opioid_use_postop',
                                      'neurop_pain_med_use_postop',
                                      'psych_comorb',
                                      'limb_side_amputation',
                                      'lvl_amputation',
                                      'pers_disord',

                                      ])

# Drop rows with missing values
df_primary = df_primary.dropna()

# Define the target variable (dependent variable) as y
X = df_primary.drop(columns=['good_outcome'])
y = df_primary['good_outcome']

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

# use cross-validation to tune the rvm model
rvm_model = EMRVC(max_iter=100000,alpha_max = 1e3)

param_grid = [
    {'kernel': ['linear']},
    {'kernel': ['rbf'], 'gamma': [0.01, 0.1, 1, 10]},
    {'kernel': ['poly'], 'degree': [2, 3, 4], 'gamma': [0.01, 0.1, 1], 'coef0': [0, 1]},
    {'kernel': ['sigmoid'], 'gamma': [0.01, 0.1, 1], 'coef0': [0, 1]},
]
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=321)

# grid_search = GridSearchCV(rvm_model, param_grid, cv=cv, n_jobs=-1, verbose=1, scoring='roc_auc')
# grid_search = GridSearchCV(rvm_model, param_grid, cv=cv, n_jobs=-1, verbose=1, scoring='f1')
grid_search = GridSearchCV(rvm_model, param_grid, cv=cv, n_jobs=-1, verbose=1, scoring='accuracy')
grid_search.fit(X_encoded, y)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"Best model: {best_model}")
print(f"Best params: {best_params}")
print(f"Best score: {best_score}")

# Create a DataFrame from the cv_results_
results_df = pd.DataFrame(grid_search.cv_results_)
results_df = results_df.sort_values(by='mean_test_score', ascending=False)
results_df = results_df.reset_index(drop=True)
results_df.to_csv(os.path.join(data_folder, "rvm_results.csv"), index=False)

# Print the best results for each kernel
kernels = results_df['param_kernel'].unique()
for kernel in kernels:
    # Filter results for the current kernel
    kernel_results = results_df[results_df['param_kernel'] == kernel]

    # Find the index of the best result for this kernel
    best_index = kernel_results['mean_test_score'].idxmax()

    # Print the best parameters and score for this kernel
    best_params = kernel_results.loc[best_index, 'params']
    best_score = kernel_results.loc[best_index, 'mean_test_score']

    print(f"Best model for kernel '{kernel}':")
    print(f"Parameters: {best_params}")
    print(f"ROC AUC Score: {best_score:.4f}")
    print("-" * 40)

'''
Best model: EMRVC(alpha_max=1000.0, gamma=0.1, init_alpha=0.00026014568158168577,
      max_iter=100000)
Best params: {'gamma': 0.1, 'kernel': 'rbf'}
Best score: 0.7888888888888889
Best model for kernel 'rbf':
Parameters: {'gamma': 0.1, 'kernel': 'rbf'}
ROC AUC Score: 0.7889
----------------------------------------
Best model for kernel 'sigmoid':
Parameters: {'coef0': 0, 'gamma': 0.01, 'kernel': 'sigmoid'}
ROC AUC Score: 0.7889
----------------------------------------
Best model for kernel 'poly':
Parameters: {'coef0': 1, 'degree': 2, 'gamma': 0.1, 'kernel': 'poly'}
ROC AUC Score: 0.7722
----------------------------------------
Best model for kernel 'linear':
Parameters: {'kernel': 'linear'}
ROC AUC Score: 0.7571
----------------------------------------
'''

