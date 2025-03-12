import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn_rvm import EMRVC
from features_name_dict import combined_cols_dict
from matplotlib import pyplot as plt
from utils import plot_auroc_with_ci

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

# Initialize the RVM model
rvm_model = EMRVC(alpha_max=1000.0, coef0=0, degree=2, gamma=1,
      init_alpha=9.611687812379854e-05, kernel='poly', max_iter=100000)

# Number of iterations for train-test splits
n_iterations = 10

# Lists to store overall performance metrics
rvm_AUROCs = []
rvm_accuracies = []
rvm_f1s = []
sensitivity_rvm_list = []
specificity_rvm_list = []

# Lists for gender-specific metrics (Male)
rvm_AUROCs_male = []
rvm_accuracies_male = []
rvm_f1s_male = []
sensitivity_rvm_male_list = []
specificity_rvm_male_list = []

# Lists for gender-specific metrics (Female)
rvm_AUROCs_female = []
rvm_accuracies_female = []
rvm_f1s_female = []
sensitivity_rvm_female_list = []
specificity_rvm_female_list = []

# Perform multiple train-test splits and compute metrics
for i in range(n_iterations):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=321 + i, stratify=y
    )

    # Train the model on the training set
    rvm_model.fit(X_train, y_train)

    # Overall predictions and metrics
    rvm_test_pred = rvm_model.predict_proba(X_test)[:, 1]
    rvm_AUROCs.append(roc_auc_score(y_test, rvm_test_pred))
    rvm_accuracies.append(rvm_model.score(X_test, y_test))
    rvm_f1s.append(f1_score(y_test, rvm_model.predict(X_test)))

    # Compute overall confusion matrix values for sensitivity and specificity
    y_pred_overall = rvm_model.predict(X_test)
    cm_overall = confusion_matrix(y_test, y_pred_overall)
    TN, FP, FN, TP = cm_overall.ravel()
    sens_overall = TP / (TP + FN) if (TP + FN) > 0 else 0
    spec_overall = TN / (TN + FP) if (TN + FP) > 0 else 0
    sensitivity_rvm_list.append(sens_overall)
    specificity_rvm_list.append(spec_overall)

    # --------------------------
    # Split the test set by gender_Male
    # --------------------------
    # Note: This assumes 'gender_Male' is a column in X_encoded.
    X_test_male = X_test[X_test['gender_Male'] == 1]
    y_test_male = y_test[X_test['gender_Male'] == 1]
    X_test_female = X_test[X_test['gender_Male'] == 0]
    y_test_female = y_test[X_test['gender_Male'] == 0]

    # Metrics for the male subgroup
    if not X_test_male.empty:
        rvm_test_pred_male = rvm_model.predict_proba(X_test_male)[:, 1]
        rvm_AUROCs_male.append(roc_auc_score(y_test_male, rvm_test_pred_male))
        rvm_accuracies_male.append(rvm_model.score(X_test_male, y_test_male))
        rvm_f1s_male.append(f1_score(y_test_male, rvm_model.predict(X_test_male)))
        y_pred_male = rvm_model.predict(X_test_male)
        cm_male = confusion_matrix(y_test_male, y_pred_male)
        TN_male, FP_male, FN_male, TP_male = cm_male.ravel()
        sens_male = TP_male / (TP_male + FN_male) if (TP_male + FN_male) > 0 else 0
        spec_male = TN_male / (TN_male + FP_male) if (TN_male + FP_male) > 0 else 0
        sensitivity_rvm_male_list.append(sens_male)
        specificity_rvm_male_list.append(spec_male)
    else:
        rvm_AUROCs_male.append(None)
        rvm_accuracies_male.append(None)
        rvm_f1s_male.append(None)
        sensitivity_rvm_male_list.append(None)
        specificity_rvm_male_list.append(None)

    # Metrics for the female subgroup
    if not X_test_female.empty:
        rvm_test_pred_female = rvm_model.predict_proba(X_test_female)[:, 1]
        rvm_AUROCs_female.append(roc_auc_score(y_test_female, rvm_test_pred_female))
        rvm_accuracies_female.append(rvm_model.score(X_test_female, y_test_female))
        rvm_f1s_female.append(f1_score(y_test_female, rvm_model.predict(X_test_female)))
        y_pred_female = rvm_model.predict(X_test_female)
        cm_female = confusion_matrix(y_test_female, y_pred_female)
        TN_female, FP_female, FN_female, TP_female = cm_female.ravel()
        sens_female = TP_female / (TP_female + FN_female) if (TP_female + FN_female) > 0 else 0
        spec_female = TN_female / (TN_female + FP_female) if (TN_female + FP_female) > 0 else 0
        sensitivity_rvm_female_list.append(sens_female)
        specificity_rvm_female_list.append(spec_female)
    else:
        rvm_AUROCs_female.append(None)
        rvm_accuracies_female.append(None)
        rvm_f1s_female.append(None)
        sensitivity_rvm_female_list.append(None)
        specificity_rvm_female_list.append(None)


# Define a helper function to compute mean and standard deviation safely
def safe_mean_std(values):
    # Filter out None values
    filtered = [v for v in values if v is not None]
    if len(filtered) == 0:
        return np.nan, np.nan
    return np.mean(filtered), np.std(filtered)


# Compute aggregated metrics (mean and std) for overall metrics
overall_auroc_mean, overall_auroc_std = safe_mean_std(rvm_AUROCs)
overall_accuracy_mean, overall_accuracy_std = safe_mean_std(rvm_accuracies)
overall_f1_mean, overall_f1_std = safe_mean_std(rvm_f1s)
overall_sens_mean, overall_sens_std = safe_mean_std(sensitivity_rvm_list)
overall_spec_mean, overall_spec_std = safe_mean_std(specificity_rvm_list)

# Compute aggregated metrics for the male subgroup
male_auroc_mean, male_auroc_std = safe_mean_std(rvm_AUROCs_male)
male_accuracy_mean, male_accuracy_std = safe_mean_std(rvm_accuracies_male)
male_f1_mean, male_f1_std = safe_mean_std(rvm_f1s_male)
male_sens_mean, male_sens_std = safe_mean_std(sensitivity_rvm_male_list)
male_spec_mean, male_spec_std = safe_mean_std(specificity_rvm_male_list)

# Compute aggregated metrics for the female subgroup
female_auroc_mean, female_auroc_std = safe_mean_std(rvm_AUROCs_female)
female_accuracy_mean, female_accuracy_std = safe_mean_std(rvm_accuracies_female)
female_f1_mean, female_f1_std = safe_mean_std(rvm_f1s_female)
female_sens_mean, female_sens_std = safe_mean_std(sensitivity_rvm_female_list)
female_spec_mean, female_spec_std = safe_mean_std(specificity_rvm_female_list)

# Combine mean and std into formatted strings using 4 decimal places
combined_overall = [
    f"{overall_auroc_mean:.4f} ± {overall_auroc_std:.4f}",
    f"{overall_accuracy_mean:.4f} ± {overall_accuracy_std:.4f}",
    f"{overall_f1_mean:.4f} ± {overall_f1_std:.4f}",
    f"{overall_sens_mean:.4f} ± {overall_sens_std:.4f}",
    f"{overall_spec_mean:.4f} ± {overall_spec_std:.4f}"
]

combined_male = [
    f"{male_auroc_mean:.4f} ± {male_auroc_std:.4f}",
    f"{male_accuracy_mean:.4f} ± {male_accuracy_std:.4f}",
    f"{male_f1_mean:.4f} ± {male_f1_std:.4f}",
    f"{male_sens_mean:.4f} ± {male_sens_std:.4f}",
    f"{male_spec_mean:.4f} ± {male_spec_std:.4f}"
]

combined_female = [
    f"{female_auroc_mean:.4f} ± {female_auroc_std:.4f}",
    f"{female_accuracy_mean:.4f} ± {female_accuracy_std:.4f}",
    f"{female_f1_mean:.4f} ± {female_f1_std:.4f}",
    f"{female_sens_mean:.4f} ± {female_sens_std:.4f}",
    f"{female_spec_mean:.4f} ± {female_spec_std:.4f}"
]

# Create a DataFrame to store the combined results
results_combined = pd.DataFrame({
    'Metric': ['AUROC', 'Accuracy', 'F1', 'Sensitivity', 'Specificity'],
    'Overall': combined_overall,
    'Male': combined_male,
    'Female': combined_female
})

# Export the combined results as a CSV file
results_csv_path = os.path.join(fig_folder, 'aggregated_test_metrics_by_gender_combined.csv')
results_combined.to_csv(results_csv_path, index=False)
print(f"Aggregated combined results exported to {results_csv_path}")
