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
from features_name_dict import combined_cols_dict

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


# Define the models
logistic_regression = LogisticRegression(C=0.34234234234234234, class_weight='balanced',
                   max_iter=1000000, penalty='l1', solver='liblinear',
                   tol=0.001)
random_forest = RandomForestClassifier(class_weight='balanced', min_samples_leaf=10,
                       n_estimators=700, random_state=321)
rvm_model = EMRVC(alpha_max=1000.0, coef0=0, degree=2, gamma=1,
      init_alpha=9.611687812379854e-05, kernel='poly', max_iter=100000)

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
Logistic Regression Mean±Std ROC AUC: 0.7908 ± 0.0703
Random Forest Mean±Std ROC AUC: 0.8102 ± 0.0880
RVM Mean±Std ROC AUC: 0.8020 ± 0.0536

Logistic Regression Mean±Std Accuracy: 0.6476 ± 0.0713
Random Forest Mean±Std Accuracy: 0.7762 ± 0.0826
RVM Mean±Std Accuracy: 0.7381 ± 0.0775

Logistic Regression Mean±Std F1 Score: 0.6876 ± 0.0848
Random Forest Mean±Std F1 Score: 0.8279 ± 0.0661
RVM Mean±Std F1 Score: 0.7990 ± 0.0771
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

# modify feature names using the imported combined_cols_dict
feature_names = [combined_cols_dict[feature] for feature in feature_names]

# Summarize feature importance using SHAP values for the positive class
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_encoded_array, feature_names=feature_names, plot_type="bar", show=False, max_display= 30)
# Save the plot
plt.tight_layout()
plt.subplots_adjust(right=1.2)
plt.savefig(os.path.join(fig_folder, 'rvm_shap_feature_importance.png'), bbox_inches='tight', dpi=300)

# Visualize SHAP feature importance as a beeswarm plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_encoded_array, feature_names=feature_names, show=False, max_display= 30)  # Beeswarm plot with feature names
# Save the plot
plt.tight_layout()
plt.savefig(os.path.join(fig_folder, 'rvm_shap_feature_importance_distribution.png'), bbox_inches='tight', dpi=300)