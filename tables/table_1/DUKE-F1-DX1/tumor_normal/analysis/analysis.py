import numpy as np
import pickle
import pathlib
from model import utils
from scipy.stats import spearmanr
path = pathlib.Path.cwd()
if path.stem == 'DeepTMB':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('DeepTMB')]
    import sys
    sys.path.append(str(cwd))

t = utils.LogTransform(bias=4, min_x=0)

##counting metrics
data = pickle.load(open(cwd / 'tables' / 'table_1' / 'DUKE-F1-DX1' / 'tumor_normal' / 'data' / 'data.pkl', 'rb'))
ancestry = pickle.load(open(cwd / 'files' / 'ethnicity.pkl', 'rb'))

[data.pop(i) for i in list(data.keys()) if not data[i]]

non_syn = ['Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins', 'In_Frame_Del', 'In_Frame_Ins', 'Nonstop_Mutation']
non_syn_data = {i: sum([data[i][5].to_dict()[j] for j in data[i][5].index if j in non_syn]) for i in data}
cutoff = np.percentile(list(non_syn_data.values()), 98)
anc = np.array([ancestry.get(i[:12], 'OA') for i in data])
anc_encoding = {'AA': 1, 'EA': 2, 'EAA': 3, 'NA': 4, 'OA': 0}
anc = np.array([anc_encoding[i] for i in anc])
mask = list(non_syn_data.values()) < cutoff
anc = anc[mask]

X = np.array([i / (j[1] / 1e6) for i, j in zip(non_syn_data.values(), data.values())])
Y = np.array([i[2] / (i[3] / 1e6) for i in data.values()])

p_TMB = X[mask]
Y = Y[mask]

print('counting')
tmb_high = p_TMB >= 5
print(round(np.mean(np.abs(Y[tmb_high] - p_TMB[tmb_high])), 2))
print(round(spearmanr(p_TMB[tmb_high], Y[tmb_high])[0], 2))

print('linear prob')
run_predictions, test_idx, values, losses = pickle.load(open(cwd / 'tables' / 'table_1' / 'DUKE-F1-DX1' / 'tumor_normal' / 'results' / 'linear_prob_nonsyn_predictions.pkl', 'rb'))
tmb_high = p_TMB[np.concatenate(test_idx)] >= 5
print(round(np.mean(np.abs(Y[np.concatenate(test_idx)][tmb_high] - t.inv(np.concatenate(run_predictions)[tmb_high]))), 2))
print(round(spearmanr(t.inv(np.concatenate(run_predictions)[tmb_high]), Y[np.concatenate(test_idx)][tmb_high])[0], 2))

print('gmm prob')
run_predictions, test_idx, values, losses = pickle.load(open(cwd / 'tables' / 'table_1' / 'DUKE-F1-DX1' / 'tumor_normal' / 'results' / 'gmm_nonsyn_predictions.pkl', 'rb'))
print(round(np.mean(np.abs(Y[np.concatenate(test_idx)][tmb_high] - t.inv(np.concatenate(run_predictions)[tmb_high]))), 2))
print(round(spearmanr(t.inv(np.concatenate(run_predictions)[tmb_high]), Y[np.concatenate(test_idx)][tmb_high])[0], 2))

print('gmm hotspot')
run_predictions, test_idx, values, losses = pickle.load(open(cwd / 'tables' / 'table_1' / 'DUKE-F1-DX1' / 'tumor_normal' / 'results' / 'gmm_hotspots_predictions.pkl', 'rb'))
print(round(np.mean(np.abs(Y[np.concatenate(test_idx)][tmb_high] - t.inv(np.concatenate(run_predictions)[tmb_high]))), 2))
print(round(spearmanr(t.inv(np.concatenate(run_predictions)[tmb_high]), Y[np.concatenate(test_idx)][tmb_high])[0], 2))

print('gmm syn')
run_predictions, test_idx, values, losses = pickle.load(open(cwd / 'tables' / 'table_1' / 'DUKE-F1-DX1' / 'tumor_normal' / 'results' / 'gmm_syn_predictions.pkl', 'rb'))
print(round(np.mean(np.abs(Y[np.concatenate(test_idx)][tmb_high] - t.inv(np.concatenate(run_predictions)[tmb_high]))), 2))
print(round(spearmanr(t.inv(np.concatenate(run_predictions)[tmb_high]), Y[np.concatenate(test_idx)][tmb_high])[0], 2))

print('gmm syn hotspots')
run_predictions, test_idx, values, losses = pickle.load(open(cwd / 'tables' / 'table_1' / 'DUKE-F1-DX1' / 'tumor_normal' / 'results' / 'gmm_syn_hotspots_predictions.pkl', 'rb'))
print(round(np.mean(np.abs(Y[np.concatenate(test_idx)][tmb_high] - t.inv(np.concatenate(run_predictions)[tmb_high]))), 2))
print(round(spearmanr(t.inv(np.concatenate(run_predictions)[tmb_high]), Y[np.concatenate(test_idx)][tmb_high])[0], 2))

print('gmm syn hotspots ancestry')
run_predictions, test_idx, values, losses = pickle.load(open(cwd / 'tables' / 'table_1' / 'DUKE-F1-DX1' / 'tumor_normal' / 'results' / 'gmm_syn_hotspots_ancestry_predictions.pkl', 'rb'))
print(round(np.mean(np.abs(Y[np.concatenate(test_idx)][tmb_high] - t.inv(np.concatenate(run_predictions)[tmb_high]))), 2))
print(round(spearmanr(t.inv(np.concatenate(run_predictions)[tmb_high]), Y[np.concatenate(test_idx)][tmb_high])[0], 2))

print('gmm multi ancestry')
run_predictions, test_idx, values, losses = pickle.load(open(cwd / 'tables' / 'table_1' / 'DUKE-F1-DX1' / 'tumor_normal' / 'results' / 'gmm_multi_ancestry_predictions.pkl', 'rb'))
print(round(np.mean(np.abs(Y[np.concatenate(test_idx)][tmb_high] - t.inv(np.concatenate(run_predictions)[tmb_high]))), 2))
print(round(spearmanr(t.inv(np.concatenate(run_predictions)[tmb_high]), Y[np.concatenate(test_idx)][tmb_high])[0], 2))

