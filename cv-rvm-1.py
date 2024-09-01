import numpy as np
from sklearn_rvm import EMRVC
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel
from sklearn.metrics import roc_auc_score

# Load the data
proj_folder = r"E:\work_local_backup\neuroma_data_project\aim_1"
data_folder = os.path.join(proj_folder, "data")
data_file = os.path.join(data_folder, "TMR_dataset_ML_March24.xlsx")

fig_folder = os.path.join(proj_folder, "figures", "secondary_TMR")
df = pd.read_excel(data_file)

df_secondary = df[df['timing_tmr'] == 'Secondary']

df_secondary = df_secondary.drop(columns=['record_id', 'participant_id',
                                          'birth_date', 'race', 'adi_natrank', 'adi_statrank',
                                          'employment_status', 'insurance', 'date_amputation',
                                          'time_preopscoretotmr', 'date_injury_amputation',
                                          'type_surg_tmr', 'time_amptmr_days', 'date_surgery_ican',
                                          'date_discharge', 'follow_up_years', 'last_score',
                                          'type_surg_rpni', 'mech_injury_amputation',
                                          'malignacy_dichotomous', 'trauma_dichotomous', 'timing_tmr',
                                          'pain_score_difference', 'MCID', 'pain_mild',
                                          'pain_disappearance', 'opioid_use_postop',
                                          'neurop_pain_med_use_postop', 'psych_comorb',
                                          'limb_side_amputation', 'lvl_amputation', 'pers_disord'])

df_secondary = df_secondary.dropna()

# Define the target variable (dependent variable) as y
X = df_secondary.drop(columns=['good_outcome'])
y = df_secondary['good_outcome']

# Separate numerical and categorical columns
numerical_cols = X.select_dtypes(include='number').columns
categorical_cols = X.select_dtypes(include='object').columns

# One-hot encode the categorical columns
onehot_encoder = OneHotEncoder(drop='first', sparse_output=False)
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

# Set up the best polynomial kernel RVM model
rvm_poly_model = EMRVC(kernel='poly', degree=2, gamma=0.01, coef0=1, max_iter=100000, alpha_max=1e3)

# Train and evaluate the polynomial model
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=321)
scores_poly = cross_val_score(rvm_poly_model, X_encoded, y, cv=cv, scoring='roc_auc')

print(f"Polynomial Kernel ROC AUC Scores: {scores_poly}")
print(f"Polynomial Kernel Mean ROC AUC Score: {np.mean(scores_poly)}")

# Compute the combined poly_rbf kernel matrix
K_poly = polynomial_kernel(X_encoded, degree=2, gamma=0.01, coef0=1)
K_rbf = rbf_kernel(X_encoded, gamma=0.01)  # Use the best gamma for RBF
K_combined = K_poly + K_rbf

# Train the RVM model on the combined kernel matrix
rvm_poly_rbf_model = EMRVC(kernel='precomputed', max_iter=100000, alpha_max=1e3)

# Evaluate the combined poly_rbf kernel model using cross-validation
scores_poly_rbf = []
for train_index, test_index in cv.split(X_encoded, y):
    # Compute the kernel matrix for the training set
    K_train = K_combined[train_index][:, train_index]

    # Compute the kernel matrix for the test set (against the training set)
    K_test = K_combined[test_index][:, train_index]

    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Fit the model on the training kernel matrix
    rvm_poly_rbf_model.fit(K_train, y_train)

    # Since the kernel is precomputed, use the model's predict_proba directly on K_test
    y_pred_prob = rvm_poly_rbf_model.predict_proba(K_test)[:, 1]

    # Compute the ROC AUC score for this fold
    score = roc_auc_score(y_test, y_pred_prob)
    scores_poly_rbf.append(score)

print(f"Poly + RBF Kernel ROC AUC Scores: {scores_poly_rbf}")
print(f"Poly + RBF Kernel Mean ROC AUC Score: {np.mean(scores_poly_rbf)}")
