import numpy as np
import pickle
import pathlib
from model import utils
import pandas as pd
import pylab as plt
import seaborn as sns
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
mask = [(i[-1] / (i[1] / 1e6)) < cutoff for i in data.values()]
anc = np.array([ancestry.get(i[:12], 'OA') for i in data])
anc_encoding = {'AA': 1, 'EA': 2, 'EAA': 3, 'NA': 4, 'OA': 0}
anc = np.array([anc_encoding[i] for i in anc])
anc = anc[mask]

X = np.array([i[-1] / (i[1] / 1e6) for i in data.values()])
Y = np.array([i[2] / (i[3] / 1e6) for i in data.values()])

p_TMB = X[mask]
Y = Y[mask]

deviations = {}

tmb_high = p_TMB >= 5
deviations['counting'] = np.abs(Y[tmb_high] - p_TMB[tmb_high])

run_predictions, test_idx, values, losses = pickle.load(open(cwd / 'tables' / 'table_1' / 'DUKE-F1-DX1' / 'tumor_only_loose' / 'results' / 'linear_prob_nonsyn_predictions.pkl', 'rb'))
tmb_high = p_TMB[np.concatenate(test_idx)] >= 5
deviations['linear_prob'] = np.abs(Y[np.concatenate(test_idx)][tmb_high] - t.inv(np.concatenate(run_predictions)[tmb_high]))

run_predictions, test_idx, values, losses = pickle.load(open(cwd / 'tables' / 'table_1' / 'DUKE-F1-DX1' / 'tumor_only_loose' / 'results' / 'gmm_nonsyn_predictions.pkl', 'rb'))
deviations['gmm prob'] = np.abs(Y[np.concatenate(test_idx)][tmb_high] - t.inv(np.concatenate(run_predictions)[tmb_high]))

run_predictions, test_idx, values, losses = pickle.load(open(cwd / 'tables' / 'table_1' / 'DUKE-F1-DX1' / 'tumor_only_loose' / 'results' / 'gmm_hotspots_predictions.pkl', 'rb'))
t = utils.LogTransform(bias=4, min_x=min([(i[-1] - i[4]) / (i[1] / 1e6) for i in data.values()]))
deviations['gmm hotspot'] = np.abs(Y[np.concatenate(test_idx)][tmb_high] - t.inv(np.concatenate(run_predictions)[tmb_high]))
t = utils.LogTransform(bias=4, min_x=0)

run_predictions, test_idx, values, losses = pickle.load(open(cwd / 'tables' / 'table_1' / 'DUKE-F1-DX1' / 'tumor_only_loose' / 'results' / 'gmm_syn_predictions.pkl', 'rb'))
deviations['gmm syn'] = np.abs(Y[np.concatenate(test_idx)][tmb_high] - t.inv(np.concatenate(run_predictions)[tmb_high]))

run_predictions, test_idx, values, losses = pickle.load(open(cwd / 'tables' / 'table_1' / 'DUKE-F1-DX1' / 'tumor_only_loose' / 'results' / 'gmm_syn_hotspots_predictions.pkl', 'rb'))
deviations['gmm syn hotspots'] = np.abs(Y[np.concatenate(test_idx)][tmb_high] - t.inv(np.concatenate(run_predictions)[tmb_high]))

run_predictions, test_idx, values, losses = pickle.load(open(cwd / 'tables' / 'table_1' / 'DUKE-F1-DX1' / 'tumor_only_loose' / 'results' / 'gmm_syn_hotspots_ancestry_predictions.pkl', 'rb'))
deviations['gmm syn hotspots ancestry'] = np.abs(Y[np.concatenate(test_idx)][tmb_high] - t.inv(np.concatenate(run_predictions)[tmb_high]))

run_predictions, test_idx, values, losses = pickle.load(open(cwd / 'tables' / 'table_1' / 'DUKE-F1-DX1' / 'tumor_only_loose' / 'results' / 'gmm_multi_ancestry_predictions.pkl', 'rb'))
deviations['gmm multi ancestry'] = np.abs(Y[np.concatenate(test_idx)][tmb_high] - t.inv(np.concatenate(run_predictions)[tmb_high]))


deviations_df = pd.DataFrame(data=deviations)

fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(top=.941,
bottom=0.11,
left=0.215,
right=.98,
hspace=0.2,
wspace=0.2)
sns.boxplot(data=deviations_df,
            medianprops={'linewidth': 0},
            meanline=True,
            showmeans=True,
            meanprops={"color": "black"},
            ax=ax,
            orient='h')
ax.set_xlabel('Absolute Deviations', fontsize=14)
ax.tick_params(length=0, width=0, axis='y', pad=10)
ax.set_yticklabels(['panel TMB\nNonsyn', 'Linear Model\nNonsyn', 'Mixture Model\nNonsyn', 'Mixture Model\nNonsyn -Hotspots',
                    'Mixture Model\nAll', 'Mixture Model\nAll-Hotspots', 'Mixture Model\nAll-Hotspots\n+Ancestry', 'Mixture Model\nAll Inputs'],
                   )
ax.set_xlim(0, 50)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_title('Tumor Only Loose', fontsize=16)
plt.savefig(cwd / 'figures' / 'supplemental_figures' / 'figure_3' / 'tumor_only_loose.pdf')