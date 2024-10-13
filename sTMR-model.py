import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn_rvm import EMRVC
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt

# Load and prepare the data
proj_folder = r"E:\work_local_backup\neuroma_data_project\aim_1"
data_folder = os.path.join(proj_folder, "data")
data_file = os.path.join(data_folder, "TMR_dataset_ML_March24.xlsx")

fig_folder = os.path.join(proj_folder, "figures","secondary_TMR")
df = pd.read_excel(data_file)

df_secondary = df[df['timing_tmr']=='Secondary']

df_secondary = df_secondary.drop(columns=['record_id',
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
                                      # 'time_amptmr_years',
                                      # 'age_ican_surgery',
                                      'pain_score_difference',
                                      'MCID',
                                      # 'preop_score',
                                      'pain_mild',
                                      'pain_disappearance',
                                      'opioid_use_postop',
                                      'neurop_pain_med_use_postop',
                                      'psych_comorb',
                                      'limb_side_amputation',
                                      'lvl_amputation',
                                      'pers_disord',
                       ])

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


# Define the models
logistic_regression = LogisticRegression(C=0.7957957957957957, class_weight='balanced',
                   max_iter=1000000, penalty='l1', solver='liblinear',
                   tol=0.001)
random_forest = RandomForestClassifier(class_weight='balanced', min_samples_leaf=10,
                       n_estimators=400, random_state=321)
rvm_model = EMRVC(alpha_max=1000.0, coef0=1, degree=2, gamma=0.1,
      init_alpha=0.00010203040506070809, kernel='poly', max_iter=100000)

# Number of iterations for train-test splits
n_iterations = 10

# Initialize lists to store the ROC AUC scores
logistic_AUROCs = []
rf_AUROCs = []
rvm_AUROCs = []

# Initialize lists to store the accuracy scores
logistic_accuracies = []
rf_accuracies = []
rvm_accuracies = []

# Initialize lists to store the F1 scores
logistic_f1s = []
rf_f1s = []
rvm_f1s = []

# initialize the permutation importance
logistic_permutation_importances = []
rf_permutation_importances = []
rvm_permutation_importances = []

# Perform multiple train-test splits
for i in range(n_iterations):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=321 + i, stratify=y)

    # Train the individual models on the entire training set
    logistic_regression.fit(X_train, y_train)
    random_forest.fit(X_train, y_train)
    rvm_model.fit(X_train, y_train)

    # Make predictions on the test set for each model
    logistic_test_pred = logistic_regression.predict_proba(X_test)[:, 1]
    rf_test_pred = random_forest.predict_proba(X_test)[:, 1]
    rvm_test_pred = rvm_model.predict_proba(X_test)[:, 1]

    # Calculate and store the AUROC scores for each individual model
    logistic_AUROCs.append(roc_auc_score(y_test, logistic_test_pred))
    rf_AUROCs.append(roc_auc_score(y_test, rf_test_pred))
    rvm_AUROCs.append(roc_auc_score(y_test, rvm_test_pred))

    # Calculate the accuracy scores
    logistic_accuracies.append(logistic_regression.score(X_test, y_test))
    rf_accuracies.append(random_forest.score(X_test, y_test))
    rvm_accuracies.append(rvm_model.score(X_test, y_test))

    # Calculate the F1 scores
    logistic_f1s.append(f1_score(y_test, logistic_regression.predict(X_test)))
    rf_f1s.append(f1_score(y_test, random_forest.predict(X_test)))
    rvm_f1s.append(f1_score(y_test, rvm_model.predict(X_test)))

    # Calculate the permutation importance on the original features
    logistic_permutation_importance = permutation_importance(logistic_regression, X_test, y_test, n_repeats=10,
                                                             random_state=321 + i)
    rf_permutation_importance = permutation_importance(random_forest, X_test, y_test, n_repeats=10,
                                                       random_state=321 + i)
    rvm_permutation_importance = permutation_importance(rvm_model, X_test, y_test, n_repeats=10, random_state=321 + i)

    # store the permutation importances
    logistic_permutation_importances.append(logistic_permutation_importance.importances_mean)
    rf_permutation_importances.append(rf_permutation_importance.importances_mean)
    rvm_permutation_importances.append(rvm_permutation_importance.importances_mean)

# Calculate and print the mean and standard deviation of the ROC AUC scores
print(f"Logistic Regression Mean±Std ROC AUC: {np.mean(logistic_AUROCs):.4f} ± {np.std(logistic_AUROCs):.4f}")
print(f"Random Forest Mean±Std ROC AUC: {np.mean(rf_AUROCs):.4f} ± {np.std(rf_AUROCs):.4f}")
print(f"RVM Mean±Std ROC AUC: {np.mean(rvm_AUROCs):.4f} ± {np.std(rvm_AUROCs):.4f}")

# Calculate and print the mean and standard deviation of the accuracy scores
print(f"Logistic Regression Mean±Std Accuracy: {np.mean(logistic_accuracies):.4f} ± {np.std(logistic_accuracies):.4f}")
print(f"Random Forest Mean±Std Accuracy: {np.mean(rf_accuracies):.4f} ± {np.std(rf_accuracies):.4f}")
print(f"RVM Mean±Std Accuracy: {np.mean(rvm_accuracies):.4f} ± {np.std(rvm_accuracies):.4f}")

# Calculate and print the mean and standard deviation of the F1 scores
print(f"Logistic Regression Mean±Std F1 Score: {np.mean(logistic_f1s):.4f} ± {np.std(logistic_f1s):.4f}")
print(f"Random Forest Mean±Std F1 Score: {np.mean(rf_f1s):.4f} ± {np.std(rf_f1s):.4f}")
print(f"RVM Mean±Std F1 Score: {np.mean(rvm_f1s):.4f} ± {np.std(rvm_f1s):.4f}")

'''
Logistic Regression Mean ROC AUC: 0.8165 ± 0.0890
Random Forest Mean ROC AUC: 0.8462 ± 0.0710
RVM Mean ROC AUC: 0.8473 ± 0.0948

Logistic Regression Mean Accuracy: 0.7050 ± 0.0757
Random Forest Mean Accuracy: 0.7500 ± 0.0806
RVM Mean Accuracy: 0.7700 ± 0.0781

Logistic Regression Mean F1 Score: 0.7500 ± 0.0731
Random Forest Mean F1 Score: 0.7968 ± 0.0722
RVM Mean F1 Score: 0.8256 ± 0.0566
'''


# the best model is the RVM model, so we will use it to calculate the feature importance using SHAP
# Import SHAP for feature importance calculation

import shap

# Convert Pandas DataFrame to NumPy array for SHAP compatibility
X_encoded_array = X_encoded.values  # This will return a NumPy array

# Train the RVM model on the entire dataset (just for clarity)
rvm_model.fit(X_encoded_array, y)

# Define a function to return only the probability for the positive class
def predict_positive_proba(X):
    return rvm_model.predict_proba(X)[:, 1]

# Initialize SHAP explainer using the wrapped function
explainer = shap.KernelExplainer(predict_positive_proba, X_encoded_array)

# Calculate SHAP values (this can be computationally intensive for large datasets)
shap_values = explainer.shap_values(X_encoded_array)

# # Debugging: Check if SHAP values and X_encoded shapes match
# print("X_encoded_array shape:", X_encoded_array.shape)
# print("shap_values[1] shape:", np.array(shap_values[1]).shape)

# Extract feature names from the X_encoded DataFrame
feature_names = X_encoded.columns

# Summarize feature importance using SHAP values for the positive class
shap.summary_plot(shap_values, X_encoded_array, feature_names=feature_names, plot_type="bar")
# Save the plot
plt.savefig(os.path.join(fig_folder, 'rvm_shap_feature_importance.png'))

# Visualize SHAP feature importance as a beeswarm plot
shap.summary_plot(shap_values, X_encoded_array, feature_names=feature_names)  # Beeswarm plot with feature names
# Save the plot
plt.savefig(os.path.join(fig_folder, 'rvm_shap_feature_importance_distribution.png'))