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
data = pickle.load(open(cwd / 'tables' / 'table_1' / 'VICC_01_R2' / 'data' / 'data.pkl', 'rb'))
nci_table = pd.read_csv(open(cwd / 'files' / 'NCI-T.tsv'), sep='\t').dropna()
nci_dict = {i: j for i, j in zip(nci_table['Tumor_Sample_Barcode'].values, nci_table['NCI-T Label TMB'].values)}

result = data.copy()
[result.pop(i) for i in data if i not in nci_dict]
values = [i for i in result.values() if i]
non_syn = ['Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins', 'In_Frame_Del', 'In_Frame_Ins', 'Nonstop_Mutation']
non_syn_counts = [sum([i[5].to_dict()[j] for j in i[5].index if j in non_syn]) for i in values]
X_non_syn = np.array([i / (j[1] / 1e6) for i, j in zip(non_syn_counts, values)])
X = np.array([i[0] / (i[1] / 1e6) for i in values])
Y = np.array([i[2] / (i[3] / 1e6) for i in values])

nci = np.array([nci_dict[i] for i in result if result[i]])

class_counts = dict(zip(*np.unique(nci, return_counts=True)))

mask = [class_counts[i] >= 50 for i in nci]
nci = nci[mask]
X_non_syn = X_non_syn[mask]
X = X[mask]
Y = Y[mask]

print('counting')
print(round(np.sqrt(np.mean((Y - X_non_syn)**2)), 2))
print(round(np.mean(np.abs(Y - X_non_syn)), 2))
print(round(r2_score(Y, X_non_syn), 2))


print('deeptmb')
run_predictions, test_idx, values = pickle.load(open(cwd / 'tables' / 'table_1' / 'VICC_01_R2' / 'results' / 'run_nonsyn_predictions.pkl', 'rb'))

print(round(np.sqrt(np.mean((np.concatenate(run_predictions) - Y[np.concatenate(test_idx)])**2)), 2))
print(round(np.mean(np.abs(np.concatenate(run_predictions) - Y[np.concatenate(test_idx)])), 2))
print(round(r2_score(Y[np.concatenate(test_idx)], np.concatenate(run_predictions)), 2))

print('hotspots')
run_cancer_predictions, test_idx, values = pickle.load(open(cwd / 'tables' / 'table_1' / 'VICC_01_R2' / 'results' / 'run_hotspots_predictions.pkl', 'rb'))

print(round(np.sqrt(np.mean((np.concatenate(run_cancer_predictions) - Y[np.concatenate(test_idx)])**2)), 2))
print(round(np.mean(np.abs(np.concatenate(run_cancer_predictions) - Y[np.concatenate(test_idx)])), 2))
print(round(r2_score(Y[np.concatenate(test_idx)], np.concatenate(run_cancer_predictions)), 2))


print('syn')
run_predictions, test_idx, values = pickle.load(open(cwd / 'tables' / 'table_1' / 'VICC_01_R2' / 'results' / 'run_predictions.pkl', 'rb'))

print(round(np.sqrt(np.mean((np.concatenate(run_predictions) - Y[np.concatenate(test_idx)])**2)), 2))
print(round(np.mean(np.abs(np.concatenate(run_predictions) - Y[np.concatenate(test_idx)])), 2))
print(round(r2_score(Y[np.concatenate(test_idx)], np.concatenate(run_predictions)), 2))

print('hotspots syn')
run_cancer_predictions, test_idx, values = pickle.load(open(cwd / 'tables' / 'table_1' / 'VICC_01_R2' / 'results' / 'run_hotspots_syn_predictions.pkl', 'rb'))

print(round(np.sqrt(np.mean((np.concatenate(run_cancer_predictions) - Y[np.concatenate(test_idx)])**2)), 2))
print(round(np.mean(np.abs(np.concatenate(run_cancer_predictions) - Y[np.concatenate(test_idx)])), 2))
print(round(r2_score(Y[np.concatenate(test_idx)], np.concatenate(run_cancer_predictions)), 2))

print('nci')
run_cancer_predictions, test_idx, values = pickle.load(open(cwd / 'tables' / 'table_1' / 'VICC_01_R2' / 'results' / 'run_nci_predictions.pkl', 'rb'))

print(round(np.sqrt(np.mean((np.concatenate(run_cancer_predictions) - Y[np.concatenate(test_idx)])**2)), 2))
print(round(np.mean(np.abs(np.concatenate(run_cancer_predictions) - Y[np.concatenate(test_idx)])), 2))
print(round(r2_score(Y[np.concatenate(test_idx)], np.concatenate(run_cancer_predictions)), 2))
