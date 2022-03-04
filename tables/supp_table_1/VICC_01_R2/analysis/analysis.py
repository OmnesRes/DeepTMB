import numpy as np
import pickle
import pandas as pd
import pathlib
from sklearn.metrics import r2_score
path = pathlib.Path.cwd()
if path.stem == 'DeepTMB':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('DeepTMB')]
    import sys
    sys.path.append(str(cwd))

##counting metrics

pcawg_maf = pickle.load(open(path / 'files' / 'pcawg_maf_table.pkl', 'rb'))
samples = pickle.load(open(path / 'files' / 'pcawg_sample_table.pkl', 'rb'))
panels = pickle.load(open(path / 'files' / 'pcawg_panel_table.pkl', 'rb'))

samples = samples.loc[samples['non_syn_counts'] > 0]

pcawg_maf = pcawg_maf.loc[pcawg_maf['VICC-01-R2'] > 0]
non_syn = ['Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins', 'In_Frame_Del', 'In_Frame_Ins', 'Nonstop_Mutation']
panel_counts = pcawg_maf[['Tumor_Sample_Barcode', 'Variant_Classification']].groupby('Tumor_Sample_Barcode').apply(lambda x: pd.Series([(x['Variant_Classification'].isin(non_syn)).sum()], index=['panel_counts']))
panel_counts.reset_index(inplace=True)
samples = pd.merge(samples, panel_counts, how='inner', left_on='aliquot_id', right_on='Tumor_Sample_Barcode')

##histology embedding
samples['cancer_code'] = samples['project_code'].apply(lambda x: x.split('-')[0])
cancer_dict = {'LICA': 'liver', 'LINC': 'liver', 'LIRI': 'liver', 'BTCA': 'liver', 'BOCA': 'bone', 'BRCA': 'breast',
               'CLLE': 'blood', 'CMDI': 'blood', 'LAML': 'blood', 'MALY': 'blood', 'EOPC': 'prostate', 'PRAD': 'prostate',
               'OV': 'ovarian', 'MELA': 'skin', 'ESAD':'orogastric', 'ORCA': 'orogastric', 'GACA': 'orogastric',
               'PBCA': 'brain', 'RECA': 'renal', 'PAEN': 'pancreas', 'PACA': 'pancreas'}

samples['cancer'] = samples['cancer_code'].apply(lambda x: cancer_dict[x])
A = samples['cancer'].astype('category')
classes = A.cat.categories.values
classes_onehot = np.eye(len(classes))[A.cat.codes]
y_strat = np.argmax(classes_onehot, axis=-1)

X = np.array([i / (panels.loc[panels['Panel'] == 'VICC-01-R2']['cds'].values[0] / 1e6) for i in samples['panel_counts'].values])
Y = np.array([i / (panels.loc[panels['Panel'] == 'CDS']['cds'].values[0] / 1e6) for i in samples['non_syn_counts'].values])

mask = Y < 40
y_strat = y_strat[mask]
X = X[mask]
Y = Y[mask]

print(round(np.sqrt(np.mean((Y - X)**2)), 2))
print(round(np.mean(np.abs(Y - X)), 2))
print(round(r2_score(Y, X), 2))
print()


run_predictions, test_idx, values = pickle.load(open(cwd / 'tables' / 'supp_table_1' / 'VICC_01_R2' / 'results' / 'run_nonsyn_predictions.pkl', 'rb'))

run_predictions = np.exp(np.concatenate(run_predictions)) - 1
print(round(np.sqrt(np.mean((run_predictions - Y[np.concatenate(test_idx)])**2)), 2))
print(round(np.mean(np.abs(run_predictions - Y[np.concatenate(test_idx)])), 2))
print(round(r2_score(Y[np.concatenate(test_idx)], run_predictions), 2))
print()

run_predictions, test_idx, values = pickle.load(open(cwd / 'tables' / 'supp_table_1' / 'VICC_01_R2' / 'results' / 'run_cds_predictions.pkl', 'rb'))
run_predictions = np.exp(np.concatenate(run_predictions)) - 1
print(round(np.sqrt(np.mean((run_predictions - Y[np.concatenate(test_idx)])**2)), 2))
print(round(np.mean(np.abs(run_predictions - Y[np.concatenate(test_idx)])), 2))
print(round(r2_score(Y[np.concatenate(test_idx)], run_predictions), 2))
print()

run_predictions, test_idx, values = pickle.load(open(cwd / 'tables' / 'supp_table_1' / 'VICC_01_R2' / 'results' / 'run_predictions.pkl', 'rb'))
run_predictions = np.exp(np.concatenate(run_predictions)) - 1
print(round(np.sqrt(np.mean((run_predictions - Y[np.concatenate(test_idx)])**2)), 2))
print(round(np.mean(np.abs(run_predictions - Y[np.concatenate(test_idx)])), 2))
print(round(r2_score(Y[np.concatenate(test_idx)], run_predictions), 2))
print()

