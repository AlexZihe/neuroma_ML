import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn_rvm import EMRVC
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt

# Load and prepare the data
# Load the data
# proj_folder = "/Users/zihealexzhang/work_local/neuroma_data_project/aim_1"
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
logistic_regression = LogisticRegression(C=7.132132132132132, max_iter=1000000, penalty='l1',
                                         solver='liblinear', tol=0.001)
random_forest = RandomForestClassifier(min_samples_leaf=4, n_estimators=500, random_state=321)
rvm_model = EMRVC(alpha_max=1000.0, coef0=1, degree=2, gamma=0.01, kernel='poly', max_iter=100000)

# Number of iterations for train-test splits
n_iterations = 50

# Initialize lists to store the ROC AUC scores
logistic_scores = []
rf_scores = []
rvm_scores = []
stacked_scores = []

# Initialize lists to store the accuracy scores
logistic_accuracies = []
rf_accuracies = []
rvm_accuracies = []
stacked_accuracies = []

# Initialize lists to store the F1 scores
logistic_f1s = []
rf_f1s = []
rvm_f1s = []
stacked_f1s = []

# initialize the permutation importance
logistic_permutation_importances = []
rf_permutation_importances = []
rvm_permutation_importances = []
stacked_features_permutation_importances = []

# Perform multiple train-test splits
for i in range(n_iterations):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=321 + i, stratify=y)

    # Generate out-of-fold predictions for each model using cross-validation
    logistic_pred = cross_val_predict(logistic_regression, X_train, y_train, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=321 + i), method='predict_proba')[:, 1]
    rf_pred = cross_val_predict(random_forest, X_train, y_train, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=321 + i), method='predict_proba')[:, 1]
    rvm_pred = cross_val_predict(rvm_model, X_train, y_train, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=321 + i), method='predict_proba')[:, 1]

    # Stack the predictions as new features
    stacked_features_train = np.column_stack((logistic_pred, rf_pred, rvm_pred))
    stacked_features_test = np.column_stack((
        logistic_regression.fit(X_train, y_train).predict_proba(X_test)[:, 1],
        random_forest.fit(X_train, y_train).predict_proba(X_test)[:, 1],
        rvm_model.fit(X_train, y_train).predict_proba(X_test)[:, 1],
    ))

    # Train the individual models on the entire training set
    logistic_regression.fit(X_train, y_train)
    random_forest.fit(X_train, y_train)
    rvm_model.fit(X_train, y_train)

    # Make predictions on the test set for each model
    logistic_test_pred = logistic_regression.predict_proba(X_test)[:, 1]
    rf_test_pred = random_forest.predict_proba(X_test)[:, 1]
    rvm_test_pred = rvm_model.predict_proba(X_test)[:, 1]

    # Calculate and store the AUROC scores for each individual model
    logistic_scores.append(roc_auc_score(y_test, logistic_test_pred))
    rf_scores.append(roc_auc_score(y_test, rf_test_pred))
    rvm_scores.append(roc_auc_score(y_test, rvm_test_pred))

    # Train the final stacked model (Logistic Regression)
    stacked_model = LogisticRegression()
    stacked_model.fit(stacked_features_train, y_train)

    # Evaluate the performance of the stacked model on the test set
    y_pred_prob = stacked_model.predict_proba(stacked_features_test)[:, 1]
    stacked_scores.append(roc_auc_score(y_test, y_pred_prob))

    # Calculate the accuracy scores
    logistic_accuracies.append(logistic_regression.score(X_test, y_test))
    rf_accuracies.append(random_forest.score(X_test, y_test))
    rvm_accuracies.append(rvm_model.score(X_test, y_test))
    stacked_accuracies.append(stacked_model.score(stacked_features_test, y_test))

    # Calculate the F1 scores
    logistic_f1s.append(f1_score(y_test, logistic_regression.predict(X_test)))
    rf_f1s.append(f1_score(y_test, random_forest.predict(X_test)))
    rvm_f1s.append(f1_score(y_test, rvm_model.predict(X_test)))
    stacked_f1s.append(f1_score(y_test, stacked_model.predict(stacked_features_test)))

    # Calculate the permutation importance on the original features
    logistic_permutation_importance = permutation_importance(logistic_regression, X_test, y_test, n_repeats=10,
                                                             random_state=321 + i)
    rf_permutation_importance = permutation_importance(random_forest, X_test, y_test, n_repeats=10,
                                                       random_state=321 + i)
    rvm_permutation_importance = permutation_importance(rvm_model, X_test, y_test, n_repeats=10, random_state=321 + i)
    stacked_features_permutation_importance = permutation_importance(stacked_model, stacked_features_test, y_test,
                                                                     n_repeats=10, random_state=321 + i)
    # store the permutation importances
    logistic_permutation_importances.append(logistic_permutation_importance.importances_mean)
    rf_permutation_importances.append(rf_permutation_importance.importances_mean)
    rvm_permutation_importances.append(rvm_permutation_importance.importances_mean)
    stacked_features_permutation_importances.append(stacked_features_permutation_importance.importances_mean)

# Calculate and print the mean and standard deviation of the ROC AUC scores
print(f"Logistic Regression Mean ROC AUC: {np.mean(logistic_scores):.4f} ± {np.std(logistic_scores):.4f}")
print(f"Random Forest Mean ROC AUC: {np.mean(rf_scores):.4f} ± {np.std(rf_scores):.4f}")
print(f"RVM Mean ROC AUC: {np.mean(rvm_scores):.4f} ± {np.std(rvm_scores):.4f}")
print(f"Final Stacked Model Mean ROC AUC: {np.mean(stacked_scores):.4f} ± {np.std(stacked_scores):.4f}")

# Calculate and print the mean and standard deviation of the accuracy scores
print(f"Logistic Regression Mean Accuracy: {np.mean(logistic_accuracies):.4f} ± {np.std(logistic_accuracies):.4f}")
print(f"Random Forest Mean Accuracy: {np.mean(rf_accuracies):.4f} ± {np.std(rf_accuracies):.4f}")
print(f"RVM Mean Accuracy: {np.mean(rvm_accuracies):.4f} ± {np.std(rvm_accuracies):.4f}")
print(f"Final Stacked Model Mean Accuracy: {np.mean(stacked_accuracies):.4f} ± {np.std(stacked_accuracies):.4f}")

# Calculate and print the mean and standard deviation of the F1 scores
print(f"Logistic Regression Mean F1 Score: {np.mean(logistic_f1s):.4f} ± {np.std(logistic_f1s):.4f}")
print(f"Random Forest Mean F1 Score: {np.mean(rf_f1s):.4f} ± {np.std(rf_f1s):.4f}")
print(f"RVM Mean F1 Score: {np.mean(rvm_f1s):.4f} ± {np.std(rvm_f1s):.4f}")
print(f"Final Stacked Model Mean F1 Score: {np.mean(stacked_f1s):.4f} ± {np.std(stacked_f1s):.4f}")

# Calculate the mean and the standard error over mean of the permutation importances
logistic_avg_permutation_importances = np.mean(logistic_permutation_importances, axis=0)
logistic_sem_permutation_importances = np.std(logistic_permutation_importances, axis=0) / np.sqrt(n_iterations)

rf_avg_permutation_importances = np.mean(rf_permutation_importances, axis=0)
rf_sem_permutation_importances = np.std(rf_permutation_importances, axis=0) / np.sqrt(n_iterations)

rvm_avg_permutation_importances = np.mean(rvm_permutation_importances, axis=0)
rvm_sem_permutation_importances = np.std(rvm_permutation_importances, axis=0) / np.sqrt(n_iterations)

stacked_features_avg_permutation_importances = np.mean(stacked_features_permutation_importances, axis=0)
stacked_features_sem_permutation_importances = np.std(stacked_features_permutation_importances, axis=0) / np.sqrt(n_iterations)

# Function to plot the feature importances
def plot_feature_importance(avg_importance, sem_importance, model_name, feature_names):
    indices = np.argsort(avg_importance)[::-1]
    sorted_avg_importance = avg_importance[indices]
    sorted_sem_importance = sem_importance[indices]
    sorted_feature_names = np.array(feature_names)[indices]

    plt.figure(figsize=(10,10))
    plt.barh(range(len(sorted_avg_importance)), sorted_avg_importance, xerr=sorted_sem_importance, capsize=5)
    plt.yticks(range(len(sorted_avg_importance)), sorted_feature_names)
    plt.title(f'{model_name} Feature Importances (Mean ± SEM)')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_folder, f'{model_name}_feature_importances.png'))
    plt.show()

# Plot the feature importances for each model
plot_feature_importance(logistic_avg_permutation_importances, logistic_sem_permutation_importances, 'Logistic Regression', X_encoded.columns)
plot_feature_importance(rf_avg_permutation_importances, rf_sem_permutation_importances, 'Random Forest', X_encoded.columns)
plot_feature_importance(rvm_avg_permutation_importances, rvm_sem_permutation_importances, 'RVM', X_encoded.columns)
plot_feature_importance(stacked_features_avg_permutation_importances, stacked_features_sem_permutation_importances, 'Final Stacked Model', ['Logistic Regression', 'Random Forest', 'RVM'])

'''
Output:

---------------------------------------------------------
AU-ROC scores for each model:

Logistic Regression Mean ROC AUC: 0.8044 ± 0.1079
Random Forest Mean ROC AUC: 0.8299 ± 0.0807
RVM Mean ROC AUC: 0.8275 ± 0.0835
Final Stacked Model Mean ROC AUC: 0.8323 ± 0.0864
---------------------------------------------------------


---------------------------------------------------------
Accuracy scores for each model:

Logistic Regression Mean Accuracy: 0.7250 ± 0.1026
Random Forest Mean Accuracy: 0.7600 ± 0.0781
RVM Mean Accuracy: 0.7700 ± 0.0755
Final Stacked Model Mean Accuracy: 0.7580 ± 0.0802
---------------------------------------------------------


---------------------------------------------------------
F1 scores for each model:

Logistic Regression Mean F1 Score: 0.7829 ± 0.0923
Random Forest Mean F1 Score: 0.8335 ± 0.0539
RVM Mean F1 Score: 0.8262 ± 0.0593
Final Stacked Model Mean F1 Score: 0.8248 ± 0.0606
---------------------------------------------------------
'''
