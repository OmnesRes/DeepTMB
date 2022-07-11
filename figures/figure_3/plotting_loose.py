from model import utils
import pickle
import pylab as plt

import pathlib
path = pathlib.Path.cwd()
if path.stem == 'DeepTMB':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('DeepTMB')]
    import sys
    sys.path.append(str(cwd))

x_pred, preds = pickle.load(open(cwd / 'figures' / 'figure_2' / 'results' / 'loose_preds.pkl', 'rb'))
t = utils.LogTransform(bias=4, min_x=0)

anc_encoding = {'AA': 1, 'EA': 2, 'EAA': 3, 'NA': 4, 'OA': 0}
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
label_names = {'EA': 'European American', 'NA': 'Native American', 'AA': 'African American', 'EAA': 'East Asian American'}
##input output plot
fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(top=.98,
bottom=0.108,
left=0.084,
right=.722,
hspace=0.2,
wspace=0.2)
for label in ['EA', 'NA', 'AA', 'EAA']:
    if label != 'OA':
        ax.plot(x_pred, preds[anc_encoding[label] - 1], linestyle='-', alpha=.7, label=label_names[label], color=colors[anc_encoding[label] - 1])
ax.hlines(t.trf(10), -.1, t.trf(40), color='k', zorder=-1000, alpha=.3)
ax.vlines(t.trf(29), -.3, t.trf(40), linestyles='dotted', colors='k', alpha=.3)
ax.set_xticks([t.trf(i) for i in [0, 2, 5, 10, 20, 40]])
ax.set_xticklabels([0, 2, 5, 10, 20, 40])
ax.set_yticks([t.trf(i) for i in [0, 2, 5, 10, 20, 40]])
ax.set_yticklabels([0, 2, 5, 10, 20, 40], fontsize=12)
ax.set_xlabel('Panel-Derived TMB', fontsize=16)
ax.set_ylabel('Exomic TMB', fontsize=16)
ax.tick_params(length=3, width=1)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_position(['outward', -3])
ax.spines['left'].set_bounds(t.trf(0), t.trf(40))
ax.spines['bottom'].set_bounds(t.trf(0), t.trf(40))
ax.set_ylim(0, t.trf(40))
ax.set_xlim(-.1, t.trf(40))
plt.legend(frameon=False, loc=(.01, .8), fontsize=10)
plt.savefig(cwd / 'figures' / 'figure_2' / 'results' / 'ancestry_loose.pdf')