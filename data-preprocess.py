import os
import pandas as pd

proj_folder = r"E:\work_local_backup\neuroma_data_project\TMR-ML"

data_folder = os.path.join(proj_folder, "data")
data_file = os.path.join(data_folder, "TMR_dataset_ML_10.16.2024.xlsx")

df = pd.read_excel(data_file)

df_primary = df[df['timing_tmr'] == 'Primary']
df_secondary = df[df['timing_tmr']=='Secondary']


pTMR_cols = ['gender', 'bmi', 'alcoholism', 'smoking', 'opioid_use_preop',
       'neurop_pain_med_use_prepo', 'diabetes', 'hypothyroidism', 'depression',
       'anxiety', 'ptsd', 'per_vasc_disease', 'ckd', 'hx_chronic_pain', 'crps',
       'distal_proximal', 'indication_amputation', 'age_amputation',
       'good_outcome']

sTMR_cols = ['gender', 'bmi', 'alcoholism', 'smoking', 'opioid_use_preop',
       'neurop_pain_med_use_prepo', 'diabetes', 'hypothyroidism', 'depression',
       'anxiety', 'ptsd', 'per_vasc_disease', 'ckd', 'hx_chronic_pain', 'crps',
       'distal_proximal', 'indication_amputation', 'time_amptmr_years',
       'age_amputation', 'age_ican_surgery', 'preop_score', 'good_outcome']

df_primary = df_primary[pTMR_cols]
df_secondary = df_secondary[sTMR_cols]

df_primary.to_csv(os.path.join(data_folder, "pTMR.csv"), index=False)
df_secondary.to_csv(os.path.join(data_folder, "sTMR.csv"), index=False)
