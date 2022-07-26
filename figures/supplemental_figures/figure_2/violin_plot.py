import pandas as pd
import pylab as plt
import seaborn as sns
import numpy as np
import pickle
import pathlib
from model import utils


path = pathlib.Path.cwd()
if path.stem == 'DeepTMB':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('DeepTMB')]
    import sys
    sys.path.append(str(cwd))

data = pickle.load(open(cwd / 'tables' / 'table_1' / 'DUKE-F1-DX1' / 'tumor_normal' / 'data' / 'data.pkl', 'rb'))

[data.pop(i) for i in list(data.keys()) if not data[i]]


non_syn = ['Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins', 'In_Frame_Del', 'In_Frame_Ins', 'Nonstop_Mutation']
non_syn_data = {i: sum([data[i][5].to_dict()[j] for j in data[i][5].index if j in non_syn]) for i in data}
cutoff = np.percentile(list(non_syn_data.values()), 98)
values = [data[i] for i in data if non_syn_data[i] < cutoff]
non_syn_counts = [i for i in non_syn_data.values() if i < cutoff]

t = utils.LogTransform(bias=4, min_x=0)
X = t.trf(np.array([i / (j[1] / 1e6) for i, j in zip(non_syn_counts, values)]))
Y = t.trf(np.array([i[2] / (i[3] / 1e6) for i in values]))

ancestry = pickle.load(open(cwd / 'files' / 'ethnicity.pkl', 'rb'))

input_tmb = pd.DataFrame({'data': X, 'type': 'Panel-Derived TMB', 'sample': [i[:12] for i in data if sum([data[i][5].to_dict()[j] for j in data[i][5].index if j in non_syn]) < cutoff]})
input_tmb['ancestry'] = input_tmb['sample'].apply(lambda x: ancestry.get(x, 'nan'))
input_tmb = input_tmb.loc[~(input_tmb['ancestry'].isin(['nan', 'OA']))]

output_tmb = pd.DataFrame({'data': Y, 'type': 'Exomic TMB', 'sample': [i[:12] for i in data if sum([data[i][5].to_dict()[j] for j in data[i][5].index if j in non_syn]) < cutoff]})
output_tmb['ancestry'] = output_tmb['sample'].apply(lambda x: ancestry.get(x, 'nan'))
output_tmb = output_tmb.loc[~(output_tmb['ancestry'].isin(['nan', 'OA']))]

counts = pd.concat([input_tmb, output_tmb], ignore_index=True)

fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(top=.98,
bottom=.105,
left=0.05,
right=.99,
hspace=0.2,
wspace=0.2)
sns.violinplot(x='ancestry', y='data', data=counts,
               split=True,
               hue='type',
               ax=ax, order=['EA', 'NA', 'AA', 'EAA'], cut=0, inner=None, legend=False, width=1)
plt.setp(ax.collections, alpha=.5)
ax.set_yticks([t.trf(i) for i in [0, 2, 5, 10, 20, 35, 60]])
ax.set_yticklabels([0, 2, 5, 10, 20, 35, 60], fontsize=12)
ax.set_xticklabels(['European\nAmerican\n',
                    'Native\nAmerican\n',
                    'African\nAmerican\n',
                    'East Asian\nAmerican\n'], fontsize=12)
ax.tick_params(which='major', length=0, axis='x', pad=10)
ax.set_ylim(t.trf(0), t.trf(60))
ax.set_ylabel('TMB', fontsize=12)
ax.set_xlabel('')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_position(('outward', -15))
plt.legend(frameon=False, loc=(.44, .95), ncol=2, title_fontproperties={'size': 12})
plt.savefig(cwd / 'figures' / 'supplemental_figures' / 'figure_2' / 'violin_plots.pdf')


