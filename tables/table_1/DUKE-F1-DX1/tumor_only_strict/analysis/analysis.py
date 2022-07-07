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
data = pickle.load(open(cwd / 'tables' / 'table_1' / 'DUKE-F1-DX1' / 'tumor_only_strict' / 'data' / 'data.pkl', 'rb'))
ancestry = pickle.load(open(cwd / 'files' / 'ethnicity.pkl', 'rb'))
germline_samples = pickle.load(open(cwd / 'files' / 'germline' / 'data' / 'germline_samples.pkl', 'rb'))

[data.pop(i) for i in list(data.keys()) if not data[i]]
[germline_samples.pop(i) for i in list(germline_samples.keys()) if germline_samples[i] < 400]
[data.pop(i) for i in list(data.keys()) if i[:12] not in germline_samples]

cutoff = np.percentile([i[-1] / (i[1] / 1e6) for i in data.values()], 98)
mask = [(i[-1] / (i[1] / 1e6)) < cutoff for i in data.values()]
anc = np.array([ancestry.get(i[:12], 'OA') for i in data])
anc_encoding = {'AA': 1, 'EA': 2, 'EAA': 3, 'NA': 4, 'OA': 0}
anc = np.array([anc_encoding[i] for i in anc])
anc = anc[mask]

X = np.array([i[-1] / (i[1] / 1e6) for i in data.values()])
Y = np.array([i[2] / (i[3] / 1e6) for i in data.values()])

p_TMB = X[mask]
Y = Y[mask]

print('counting')
tmb_high = p_TMB >= 5
print(round(np.mean(np.abs(Y[tmb_high] - p_TMB[tmb_high])), 2))
print(round(spearmanr(p_TMB[tmb_high], Y[tmb_high])[0], 2))

print('linear prob')
run_predictions, test_idx, values, losses = pickle.load(open(cwd / 'tables' / 'table_1' / 'DUKE-F1-DX1' / 'tumor_only_strict' / 'results' / 'linear_prob_nonsyn_predictions.pkl', 'rb'))
tmb_high = p_TMB[np.concatenate(test_idx)] >= 5
print(round(np.mean(np.abs(Y[np.concatenate(test_idx)][tmb_high] - t.inv(np.concatenate(run_predictions)[tmb_high]))), 2))
print(round(spearmanr(t.inv(np.concatenate(run_predictions)[tmb_high]), Y[np.concatenate(test_idx)][tmb_high])[0], 2))

print('gmm prob')
run_predictions, test_idx, values, losses = pickle.load(open(cwd / 'tables' / 'table_1' / 'DUKE-F1-DX1' / 'tumor_only_strict' / 'results' / 'gmm_nonsyn_predictions.pkl', 'rb'))
print(round(np.mean(np.abs(Y[np.concatenate(test_idx)][tmb_high] - t.inv(np.concatenate(run_predictions)[tmb_high]))), 2))
print(round(spearmanr(t.inv(np.concatenate(run_predictions)[tmb_high]), Y[np.concatenate(test_idx)][tmb_high])[0], 2))

print('gmm hotspot')
run_predictions, test_idx, values, losses = pickle.load(open(cwd / 'tables' / 'table_1' / 'DUKE-F1-DX1' / 'tumor_only_strict' / 'results' / 'gmm_hotspots_predictions.pkl', 'rb'))
t = utils.LogTransform(bias=4, min_x=min([(i[-1] - i[4]) / (i[1] / 1e6) for i in data.values()]))
print(round(np.mean(np.abs(Y[np.concatenate(test_idx)][tmb_high] - t.inv(np.concatenate(run_predictions)[tmb_high]))), 2))
print(round(spearmanr(t.inv(np.concatenate(run_predictions)[tmb_high]), Y[np.concatenate(test_idx)][tmb_high])[0], 2))

t = utils.LogTransform(bias=4, min_x=0)

print('gmm syn')
run_predictions, test_idx, values, losses = pickle.load(open(cwd / 'tables' / 'table_1' / 'DUKE-F1-DX1' / 'tumor_only_strict' / 'results' / 'gmm_syn_predictions.pkl', 'rb'))
print(round(np.mean(np.abs(Y[np.concatenate(test_idx)][tmb_high] - t.inv(np.concatenate(run_predictions)[tmb_high]))), 2))
print(round(spearmanr(t.inv(np.concatenate(run_predictions)[tmb_high]), Y[np.concatenate(test_idx)][tmb_high])[0], 2))

print('gmm syn hotspots')
run_predictions, test_idx, values, losses = pickle.load(open(cwd / 'tables' / 'table_1' / 'DUKE-F1-DX1' / 'tumor_only_strict' / 'results' / 'gmm_syn_hotspots_predictions.pkl', 'rb'))
print(round(np.mean(np.abs(Y[np.concatenate(test_idx)][tmb_high] - t.inv(np.concatenate(run_predictions)[tmb_high]))), 2))
print(round(spearmanr(t.inv(np.concatenate(run_predictions)[tmb_high]), Y[np.concatenate(test_idx)][tmb_high])[0], 2))

print('gmm syn hotspots ancestry')
run_predictions, test_idx, values, losses = pickle.load(open(cwd / 'tables' / 'table_1' / 'DUKE-F1-DX1' / 'tumor_only_strict' / 'results' / 'gmm_syn_hotspots_ancestry_predictions.pkl', 'rb'))
print(round(np.mean(np.abs(Y[np.concatenate(test_idx)][tmb_high] - t.inv(np.concatenate(run_predictions)[tmb_high]))), 2))
print(round(spearmanr(t.inv(np.concatenate(run_predictions)[tmb_high]), Y[np.concatenate(test_idx)][tmb_high])[0], 2))

print('gmm multi ancestry')
run_predictions, test_idx, values, losses = pickle.load(open(cwd / 'tables' / 'table_1' / 'DUKE-F1-DX1' / 'tumor_only_strict' / 'results' / 'gmm_multi_ancestry_predictions.pkl', 'rb'))
print(round(np.mean(np.abs(Y[np.concatenate(test_idx)][tmb_high] - t.inv(np.concatenate(run_predictions)[tmb_high]))), 2))
print(round(spearmanr(t.inv(np.concatenate(run_predictions)[tmb_high]), Y[np.concatenate(test_idx)][tmb_high])[0], 2))
