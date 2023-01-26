import matplotlib
matplotlib.use('TKAgg')
from matplotlib import pyplot as plt
import numpy as np
from figures.figure_3.kaplan_tools import *
import pickle
from lifelines.statistics import logrank_test
from matplotlib import lines

import pathlib
path = pathlib.Path.cwd()
if path.stem == 'DeepTMB':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('DeepTMB')]
    import sys
    sys.path.append(str(cwd))

data = pickle.load(open(cwd / 'tables' / 'table_1' / 'DUKE-F1-DX1' / 'tumor_only_strict' / 'data' / 'data.pkl', 'rb'))
germline_samples = pickle.load(open(cwd / 'files' / 'germline' / 'data' / 'germline_samples.pkl', 'rb'))
samples = pickle.load(open(cwd / 'files' / 'tcga_public_sample_table.pkl', 'rb'))

[data.pop(i) for i in list(data.keys()) if not data[i]]
[germline_samples.pop(i) for i in list(germline_samples.keys()) if germline_samples[i] < 400]
[data.pop(i) for i in list(data.keys()) if i[:12] not in germline_samples]

tmb = {i[:12]: data[i][2] / (data[i][3] / 1e6) for i in data}

samples['tmb'] = samples['bcr_patient_barcode'].apply(lambda x: tmb.get(x, np.nan))
samples.dropna(subset=['tmb'], inplace=True)
df = samples.loc[samples['type'] == 'BLCA']

tmb = df['tmb'].values
times = df['OS.time'].values.astype(np.int16)[np.argsort(tmb)]
events = df['OS'].values.astype(np.int16)[np.argsort(tmb)]
tmb = np.sort(tmb)

stats = []
offset = 25
for index, cutoff in enumerate(tmb[offset: -offset]):
    stats.append(logrank_test(times[: index + offset + 1], times[index + offset + 1:], event_observed_A=events[: index + offset + 1], event_observed_B=events[index + offset + 1:]).test_statistic)

plt.scatter(tmb[offset: -offset], stats)

cutoff = np.argmax(stats)

p_value = logrank_test(times[: cutoff + offset + 1], times[cutoff + offset + 1:], event_observed_A=events[: cutoff + offset + 1], event_observed_B=events[cutoff + offset + 1:]).p_value

fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(bottom=.099)
fig.subplots_adjust(top=.98)
fig.subplots_adjust(left=.099)
fig.subplots_adjust(right=.969)

survtimes = list(zip(times[:cutoff + offset + 1], events[: cutoff + offset + 1]))
survtimes.sort(key=lambda x: (x[0], x[1]))
k_plot = kaplan(survtimes)

width = 1.4
start = 0
for i in k_plot[0]:
    ax.hlines(i[1] * 100, start, i[0], linewidths=width, color='b')
    start = i[0]

if k_plot[-1][-1][0] > k_plot[0][-1][0]:
    ax.hlines(k_plot[-1][-1][1] * 100, k_plot[0][-1][0], k_plot[-1][-1][0], linewidths=width, color='b')

for i in k_plot[1]:
    ax.vlines(i[0], i[2] * 100 - (width / 6), i[1] * 100 + (width / 6), linewidths=width, color='b')

for i in k_plot[2]:
    ax.vlines(i[0], (i[1] - .01) * 100, (i[1] + .01) * 100, linewidths=width / 2, color='b')

survtimes = list(zip(times[cutoff + offset + 1:], events[cutoff + offset + 1:]))
survtimes.sort(key=lambda x: (x[0], x[1]))
k_plot = kaplan(survtimes)
start = 0
for i in k_plot[0]:
    ax.hlines(i[1] * 100, start, i[0], linewidths=width, color='r')
    start = i[0]

if k_plot[-1][-1][0] > k_plot[0][-1][0]:
    ax.hlines(k_plot[-1][-1][1] * 100, k_plot[0][-1][0], k_plot[-1][-1][0], linewidths=width, color='r')

for i in k_plot[1]:
    ax.vlines(i[0], i[2] * 100 - (width / 6), i[1] * 100 + (width / 6), linewidths=width, color='r')

for i in k_plot[2]:
    ax.vlines(i[0], (i[1] - .01) * 100, (i[1] + .01) * 100, linewidths=width / 2, color='r')


ax.tick_params(axis='x', length=7, width=width, direction='out', labelsize=12)
ax.tick_params(axis='y', length=7, width=width, direction='out', labelsize=12)
ax.set_xlabel('Days', fontsize=12)
ax.set_ylabel('% Surviving', fontsize=12)
ax.spines['bottom'].set_position(['outward', 10])
ax.spines['left'].set_position(['outward', 10])
ax.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(width)
ax.spines['bottom'].set_linewidth(width)
ax.spines['left'].set_bounds(0, 100)
ax.spines['bottom'].set_bounds(0, 5000)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

rect1 = lines.Line2D([], [], marker="_",  linewidth=3.5, color='b')
rect2 = lines.Line2D([], [], marker="_",  linewidth=3.5, color='r')

ax.legend((rect1, rect2),
          ('  Low \n N=' + str(len(times[:cutoff + offset + 1])),
           '  High \n N=' + str(len(times[cutoff + offset + 1:]))),
          loc=(.65, .9),
          frameon=False,
          borderpad=0,
          handletextpad=.5,
          labelspacing=.3,
          fontsize=10,
          ncol=2)
ax.text(.0, .97, 'Logrank p-value=%.1E' % p_value, transform=ax.transAxes)

plt.ylim(0, 105)
plt.xlim(0, max(times))


print(logrank_test(times[: cutoff + offset + 1], times[cutoff + offset + 1:], event_observed_A=events[: cutoff + offset + 1], event_observed_B=events[cutoff + offset + 1:]).test_statistic)
print(logrank_test(times[: cutoff + offset + 1], times[cutoff + offset + 1:], event_observed_A=events[: cutoff + offset + 1], event_observed_B=events[cutoff + offset + 1:]).p_value)
print(np.mean(tmb[cutoff + offset + 1: cutoff + offset + 3]))
