import numpy as np
import pickle
import pathlib
from model import utils
from sklearn.metrics import r2_score
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

cutoff = np.percentile([(i[0] - i[4]) / (i[1] / 1e6) for i in data.values()], 98)
anc = np.array([ancestry.get(i[:12], 'OA') for i in data if ((data[i][0] - data[i][4]) / (data[i][1] / 1e6)) < cutoff])
anc_encoding = {'AA': 1, 'EA': 2, 'EAA': 3, 'NA': 4, 'OA': 0}
anc = np.array([anc_encoding[i] for i in anc])
values = [i for i in data.values() if ((i[0] - i[4]) / (i[1] / 1e6)) < cutoff]
Y = np.array([i[2] / (i[3] / 1e6) for i in values])

for i in range(1, 5):
    print(i)
    mask = anc[np.concatenate(test_idx)] == i
    print(round(np.mean(-np.concatenate(losses)[mask]), 2))
    print(round(np.sqrt(np.mean((t.inv(np.concatenate(run_predictions)[mask]) - Y[np.concatenate(test_idx)][mask])**2)), 2))
    print(round(r2_score(Y[np.concatenate(test_idx)][mask], t.inv(np.concatenate(run_predictions)[mask])), 2))
    print(round(sum((t.inv(np.concatenate(run_predictions)[mask]) >= 10) & (Y[np.concatenate(test_idx)][mask] >= 10)) / sum(Y[np.concatenate(test_idx)][mask] >= 10), 2))
    print(round(sum((t.inv(np.concatenate(run_predictions)[mask]) >= 10) & (Y[np.concatenate(test_idx)][mask] >= 10))
                / (sum((t.inv(np.concatenate(run_predictions)[mask]) >= 10) & (Y[np.concatenate(test_idx)][mask] >= 10)) + sum((t.inv(np.concatenate(run_predictions)[mask]) >= 10) & (Y[np.concatenate(test_idx)[mask]] < 10))), 2))



print('with ancestry')
run_predictions, test_idx, values, losses = pickle.load(open(cwd / 'tables' / 'table_2' / 'DUKE-F1-DX1' / 'tumor_normal' / 'results' / 'with_ancestry.pkl', 'rb'))

for i in range(1, 5):
    print(i)
    mask = anc[np.concatenate(test_idx)] == i
    print(round(np.mean(-np.concatenate(losses)[mask]), 2))
    print(round(np.sqrt(np.mean((t.inv(np.concatenate(run_predictions)[mask]) - Y[np.concatenate(test_idx)][mask])**2)), 2))
    print(round(r2_score(Y[np.concatenate(test_idx)][mask], t.inv(np.concatenate(run_predictions)[mask])), 2))
    print(round(sum((t.inv(np.concatenate(run_predictions)[mask]) >= 10) & (Y[np.concatenate(test_idx)][mask] >= 10)) / sum(Y[np.concatenate(test_idx)][mask] >= 10), 2))
    print(round(sum((t.inv(np.concatenate(run_predictions)[mask]) >= 10) & (Y[np.concatenate(test_idx)][mask] >= 10))
                / (sum((t.inv(np.concatenate(run_predictions)[mask]) >= 10) & (Y[np.concatenate(test_idx)][mask] >= 10)) + sum((t.inv(np.concatenate(run_predictions)[mask]) >= 10) & (Y[np.concatenate(test_idx)[mask]] < 10))), 2))


