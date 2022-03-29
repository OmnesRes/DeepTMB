import numpy as np
import pickle
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
data = pickle.load(open(cwd / 'tables' / 'table_1' / 'MSK_468' / 'data' / 'data.pkl', 'rb'))
sample_table = pickle.load(open(cwd / 'files' / 'tcga_public_sample_table.pkl', 'rb'))

nci_dict = {i: j for i, j in zip(sample_table['Tumor_Sample_Barcode'].values, sample_table['NCIt_tmb_label'].values) if j}

[data.pop(i) for i in list(data.keys()) if not data[i]]
[data.pop(i) for i in list(data.keys()) if i not in nci_dict]

values = [i for i in data.values() if (i[2] / (i[3] / 1e6)) <= 40]
nci = np.array([nci_dict[i] for i in data if (data[i][2] / (data[i][3] / 1e6)) <= 40])

non_syn = ['Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins', 'In_Frame_Del', 'In_Frame_Ins', 'Nonstop_Mutation']
non_syn_counts = [sum([i[5].to_dict()[j] for j in i[5].index if j in non_syn]) for i in values]
X_non_syn = np.array([i / (j[1] / 1e6) for i, j in zip(non_syn_counts, values)])
X = np.array([i[0] / (i[1] / 1e6) for i in values])
Y = np.array([i[2] / (i[3] / 1e6) for i in values])

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
print(round(sum((Y >= 10) & (X_non_syn >= 10)) / sum(X_non_syn >= 10), 2))


print('deeptmb')
run_predictions, test_idx, values = pickle.load(open(cwd / 'tables' / 'table_1' / 'MSK_468' / 'results' / 'run_nonsyn_predictions.pkl', 'rb'))

print(round(np.sqrt(np.mean((np.concatenate(run_predictions) - Y[np.concatenate(test_idx)])**2)), 2))
print(round(np.mean(np.abs(np.concatenate(run_predictions) - Y[np.concatenate(test_idx)])), 2))
print(round(r2_score(Y[np.concatenate(test_idx)], np.concatenate(run_predictions)), 2))
print(round(sum((np.concatenate(run_predictions) >= 10) & (Y[np.concatenate(test_idx)] >= 10)) / sum(np.concatenate(run_predictions) >= 10), 2))


print('hotspots')
run_cancer_predictions, test_idx, values = pickle.load(open(cwd / 'tables' / 'table_1' / 'MSK_468' / 'results' / 'run_hotspots_predictions.pkl', 'rb'))

print(round(np.sqrt(np.mean((np.concatenate(run_cancer_predictions) - Y[np.concatenate(test_idx)])**2)), 2))
print(round(np.mean(np.abs(np.concatenate(run_cancer_predictions) - Y[np.concatenate(test_idx)])), 2))
print(round(r2_score(Y[np.concatenate(test_idx)], np.concatenate(run_cancer_predictions)), 2))
print(round(sum((np.concatenate(run_predictions) >= 10) & (Y[np.concatenate(test_idx)] >= 10)) / sum(np.concatenate(run_predictions) >= 10), 2))


print('syn')
run_predictions, test_idx, values = pickle.load(open(cwd / 'tables' / 'table_1' / 'MSK_468' / 'results' / 'run_predictions.pkl', 'rb'))

print(round(np.sqrt(np.mean((np.concatenate(run_predictions) - Y[np.concatenate(test_idx)])**2)), 2))
print(round(np.mean(np.abs(np.concatenate(run_predictions) - Y[np.concatenate(test_idx)])), 2))
print(round(r2_score(Y[np.concatenate(test_idx)], np.concatenate(run_predictions)), 2))
print(round(sum((np.concatenate(run_predictions) >= 10) & (Y[np.concatenate(test_idx)] >= 10)) / sum(np.concatenate(run_predictions) >= 10), 2))


print('hotspots syn')
run_cancer_predictions, test_idx, values = pickle.load(open(cwd / 'tables' / 'table_1' / 'MSK_468' / 'results' / 'run_hotspots_syn_predictions.pkl', 'rb'))

print(round(np.sqrt(np.mean((np.concatenate(run_cancer_predictions) - Y[np.concatenate(test_idx)])**2)), 2))
print(round(np.mean(np.abs(np.concatenate(run_cancer_predictions) - Y[np.concatenate(test_idx)])), 2))
print(round(r2_score(Y[np.concatenate(test_idx)], np.concatenate(run_cancer_predictions)), 2))
print(round(sum((np.concatenate(run_predictions) >= 10) & (Y[np.concatenate(test_idx)] >= 10)) / sum(np.concatenate(run_predictions) >= 10), 2))


print('nci')
run_cancer_predictions, test_idx, values = pickle.load(open(cwd / 'tables' / 'table_1' / 'MSK_468' / 'results' / 'run_nci_predictions.pkl', 'rb'))

print(round(np.sqrt(np.mean((np.concatenate(run_cancer_predictions) - Y[np.concatenate(test_idx)])**2)), 2))
print(round(np.mean(np.abs(np.concatenate(run_cancer_predictions) - Y[np.concatenate(test_idx)])), 2))
print(round(r2_score(Y[np.concatenate(test_idx)], np.concatenate(run_cancer_predictions)), 2))
print(round(sum((np.concatenate(run_predictions) >= 10) & (Y[np.concatenate(test_idx)] >= 10)) / sum(np.concatenate(run_predictions) >= 10), 2))
