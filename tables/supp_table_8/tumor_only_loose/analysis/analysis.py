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
data = pickle.load(open(cwd / 'tables' / 'table_1' / 'DUKE-F1-DX1' / 'tumor_only_loose' / 'data' / 'data.pkl', 'rb'))
ancestry = pickle.load(open(cwd / 'files' / 'ethnicity.pkl', 'rb'))
germline_samples = pickle.load(open(cwd / 'files' / 'germline' / 'data' / 'germline_samples.pkl', 'rb'))

[data.pop(i) for i in list(data.keys()) if not data[i]]
[germline_samples.pop(i) for i in list(germline_samples.keys()) if germline_samples[i] < 400]
[data.pop(i) for i in list(data.keys()) if i[:12] not in germline_samples]

cutoff = np.percentile([i[-1] / (i[1] / 1e6) for i in data.values()], 98)
values = [i for i in data.values() if (i[-1] / (i[1] / 1e6)) < cutoff]
anc = np.array([ancestry.get(i[:12], 'OA') for i in data if (data[i][-1] / (data[i][1] / 1e6)) < cutoff])
anc_encoding = {'AA': 1, 'EA': 2, 'EAA': 3, 'NA': 4, 'OA': 0}
anc = np.array([anc_encoding[i] for i in anc])
p_TMB = np.array([i[-1] / (i[1] / 1e6) for i in values])
Y = np.array([i[2] / (i[3] / 1e6) for i in values])


print('without ancestry')
run_predictions, values, losses = pickle.load(open(cwd / 'tables' / 'supp_table_6' / 'tumor_only_loose' / 'results' / 'without_ancestry.pkl', 'rb'))


for i in range(1, 5):
    print(i)
    mask = anc == i
    tmb_high = p_TMB[mask] >= 5
    print(round(np.mean(np.abs(Y[mask][tmb_high] - t.inv(run_predictions[mask][tmb_high]))), 2))
    print(round(spearmanr(t.inv(run_predictions[mask][tmb_high]), Y[mask][tmb_high])[0], 2))


print('with ancestry')
run_predictions, values, losses = pickle.load(open(cwd / 'tables' / 'supp_table_8' / 'tumor_only_loose' / 'results' / 'with_ancestry.pkl', 'rb'))

for i in range(1, 5):
    print(i)
    mask = anc == i
    tmb_high = p_TMB[mask] >= 5
    print(round(np.mean(np.abs(Y[mask][tmb_high] - t.inv(run_predictions[mask][tmb_high]))), 2))
    print(round(spearmanr(t.inv(run_predictions[mask][tmb_high]), Y[mask][tmb_high])[0], 2))