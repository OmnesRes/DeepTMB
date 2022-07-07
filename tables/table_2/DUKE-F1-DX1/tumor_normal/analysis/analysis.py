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

data = pickle.load(open(cwd / 'tables' / 'table_1' / 'DUKE-F1-DX1' / 'tumor_normal' / 'data' / 'data.pkl', 'rb'))
ancestry = pickle.load(open(cwd / 'files' / 'ethnicity.pkl', 'rb'))

[data.pop(i) for i in list(data.keys()) if not data[i]]

print('without_ancestry')
run_predictions, test_idx, values, losses = pickle.load(open(cwd / 'tables' / 'table_2' / 'DUKE-F1-DX1' / 'tumor_normal' / 'results' / 'without_ancestry.pkl', 'rb'))

non_syn = ['Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins', 'In_Frame_Del', 'In_Frame_Ins', 'Nonstop_Mutation']
non_syn_data = {i: sum([data[i][5].to_dict()[j] for j in data[i][5].index if j in non_syn]) for i in data}
cutoff = np.percentile(list(non_syn_data.values()), 98)
anc = np.array([ancestry.get(i[:12], 'OA') for i in data])
anc_encoding = {'AA': 1, 'EA': 2, 'EAA': 3, 'NA': 4, 'OA': 0}
anc = np.array([anc_encoding[i] for i in anc])
mask = list(non_syn_data.values()) < cutoff
anc = anc[mask]
p_TMB = np.array([sum([data[i][5].to_dict()[j] for j in data[i][5].index if j in non_syn]) / (data[i][1] / 1e6) for i in data])
p_TMB = p_TMB[mask]
Y = np.array([i[2] / (i[3] / 1e6) for i in data.values()])
Y = Y[mask]


for i in range(1, 5):
    print(i)
    mask = anc[np.concatenate(test_idx)] == i
    tmb_high = p_TMB[np.concatenate(test_idx)][mask] >= 5
    print(round(np.mean(np.abs(Y[np.concatenate(test_idx)][mask][tmb_high] - t.inv(np.concatenate(run_predictions)[mask][tmb_high]))), 2))
    print(round(spearmanr(t.inv(np.concatenate(run_predictions)[mask][tmb_high]), Y[np.concatenate(test_idx)][mask][tmb_high])[0], 2))


print('with ancestry')
run_predictions, test_idx, values, losses = pickle.load(open(cwd / 'tables' / 'table_2' / 'DUKE-F1-DX1' / 'tumor_normal' / 'results' / 'with_ancestry.pkl', 'rb'))

for i in range(1, 5):
    print(i)
    mask = anc[np.concatenate(test_idx)] == i
    tmb_high = p_TMB[np.concatenate(test_idx)][mask] >= 5
    print(round(np.mean(np.abs(Y[np.concatenate(test_idx)][mask][tmb_high] - t.inv(np.concatenate(run_predictions)[mask][tmb_high]))), 2))
    print(round(spearmanr(t.inv(np.concatenate(run_predictions)[mask][tmb_high]), Y[np.concatenate(test_idx)][mask][tmb_high])[0], 2))


