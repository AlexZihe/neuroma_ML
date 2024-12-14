from matplotlib import pyplot as plt
import os
import numpy as np
from scipy.stats import sem
from sklearn.metrics import roc_curve


# Function to plot AUROC with confidence interval
def plot_auroc_with_ci(y_test_list, pred_list, model_name, fig_folder):
    plt.figure(figsize=(8, 6))
    tprs = []
    base_fpr = np.linspace(0, 1, 100)

    for y_test, preds in zip(y_test_list, pred_list):
        fpr, tpr, _ = roc_curve(y_test, preds)
        tprs.append(np.interp(base_fpr, fpr, tpr))
        tprs[-1][0] = 0.0

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)
    ci = sem(tprs, axis=0) * 1.96  # 95% confidence interval

    plt.plot(base_fpr, mean_tpr, label=f'{model_name} (mean)', color='blue')
    plt.fill_between(base_fpr, mean_tpr - ci, mean_tpr + ci, color='blue', alpha=0.2, label='95% CI')

    plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Chance')

    plt.title(f'AUROC Curve with 95% CI ({model_name})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save the plot
    save_path = os.path.join(fig_folder, f'{model_name}_auroc_with_ci.png')
    plt.savefig(save_path, dpi = 300)
    plt.close()