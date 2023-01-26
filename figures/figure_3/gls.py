import matplotlib
matplotlib.use('TKAgg')
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from model.model import Encoders, NN
from model import utils
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

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[2], True)
tf.config.experimental.set_visible_devices(physical_devices[2], 'GPU')

def make_colormap(colors):
    from matplotlib.colors import LinearSegmentedColormap, ColorConverter
    z = np.sort(list(colors.keys()))
    anchors = (z - min(z)) / (max(z) - min(z))
    CC = ColorConverter()
    R, G, B = [], [], []
    for i in range(len(z)):
        Ci = colors[z[i]]
        RGB = CC.to_rgb(Ci)
        R.append(RGB[0])
        G.append(RGB[1])
        B.append(RGB[2])
    cmap_dict = {}
    cmap_dict['red'] = [(anchors[i], R[i], R[i]) for i in range(len(R))]
    cmap_dict['green'] = [(anchors[i], G[i], G[i]) for i in range(len(G))]
    cmap_dict['blue'] = [(anchors[i], B[i], B[i]) for i in range(len(B))]
    mymap = LinearSegmentedColormap('mymap', cmap_dict)
    return mymap


mygreen = make_colormap({0: '#ffffff', .3: '#ccffcc', .6: '#99ff99', .8: '#00ff00', 1: '#00cc00'})

data = pickle.load(open(cwd / 'tables' / 'table_1' / 'DUKE-F1-DX1' / 'tumor_only_strict' / 'data' / 'data.pkl', 'rb'))
germline_samples = pickle.load(open(cwd / 'files' / 'germline' / 'data' / 'germline_samples.pkl', 'rb'))
samples = pickle.load(open(cwd / 'files' / 'tcga_public_sample_table.pkl', 'rb'))

[data.pop(i) for i in list(data.keys()) if not data[i]]
[germline_samples.pop(i) for i in list(germline_samples.keys()) if germline_samples[i] < 400]
[data.pop(i) for i in list(data.keys()) if i[:12] not in germline_samples]

samples['panel_tmb'] = samples['Tumor_Sample_Barcode'].apply(lambda x: data[x][-1] / (data[x][1] / 1e6) if x in data else np.nan)
samples['tmb'] = samples['Tumor_Sample_Barcode'].apply(lambda x: data[x][2] / (data[x][3] / 1e6) if x in data else np.nan)
samples.dropna(subset=['tmb'], inplace=True)
training_df = samples.loc[samples['type'] != 'BLCA']
test_df = samples.loc[samples['type'] == 'BLCA']

cutoff = np.percentile(training_df['panel_tmb'].values, 98)
mask = training_df['panel_tmb'].values < cutoff
X = training_df['panel_tmb'].values[mask]
Y = training_df['tmb'].values[mask]
y_weights = np.ones_like(Y)

t = utils.LogTransform(bias=4, min_x=0)
X = t.trf(X[:, np.newaxis])
Y = t.trf(Y)
X_loader = utils.Map.PassThrough(X)
Y_loader = utils.Map.PassThrough(Y)
W_loader = utils.Map.PassThrough(y_weights)

count_encoder = Encoders.Encoder(shape=(1,), layers=())
net = NN(encoders=[count_encoder.model], layers=(), mode='tfp_linear_regresion')

net.model.compile(loss=utils.log_prob_loss,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.001, patience=30, mode='min', restore_best_weights=True)]

idx_train = np.arange(len(X))
ds_train = tf.data.Dataset.from_tensor_slices((idx_train, ))
ds_train = ds_train.shuffle(buffer_size=len(idx_train), reshuffle_each_iteration=True).repeat().batch(batch_size=int(len(idx_train) * .75), drop_remainder=True)

ds_train = ds_train.map(lambda x: ((
                                    X_loader(x),
                                    ),
                                   (Y_loader(x),
                                    ),
                                    W_loader(x)
                                   )
                        ).map(utils.rescale_batch_weights)

net.model.fit(ds_train,
              steps_per_epoch=10,
              epochs=10000,
              callbacks=callbacks
              )
x_pred = np.linspace(np.min(X), np.max(t.trf(60)), 400)
y_pred = np.linspace(0, np.max(Y + .5), 1000)
Z = net.model(x_pred).log_prob(y_pred[:, np.newaxis]).numpy()
zz = np.exp(Z - Z.max(axis=0, keepdims=True))
X_pred, Y_pred = np.meshgrid(x_pred, y_pred)

means = net.model(x_pred).mean()
quantiles = [y_pred[np.argmin(np.diff((np.exp(net.model(x_pred).log_cdf(y_pred[:, np.newaxis]).numpy()) < 0.05).astype(int), axis=0), axis=0)],
             y_pred[np.argmin(np.diff((np.exp(net.model(x_pred).log_cdf(y_pred[:, np.newaxis]).numpy()) < 0.5).astype(int), axis=0), axis=0)],
             y_pred[np.argmin(np.diff((np.exp(net.model(x_pred).log_cdf(y_pred[:, np.newaxis]).numpy()) < 0.95).astype(int), axis=0), axis=0)]]
medians = y_pred[np.argmin(np.diff((np.exp(net.model(X).log_cdf(y_pred[:, np.newaxis]).numpy()) < 0.5).astype(int), axis=0), axis=0)]

upper_x = x_pred[np.argmin(np.abs(quantiles[0] - t.trf(6.73)))]
lower_x = x_pred[np.argmin(np.abs(quantiles[-1] - t.trf(6.73)))]

##input output plot
fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(top=.98,
bottom=0.108,
left=0.1,
right=.987,
hspace=0.2,
wspace=0.2)

pcm = ax.pcolormesh(X_pred, Y_pred, zz, cmap=mygreen, edgecolor='face', linewidth=0, vmax=1, vmin=0, alpha=.4)
ax.scatter(t.trf(test_df['panel_tmb'].values), t.trf(test_df['tmb'].values), s=7, edgecolor='none', alpha=1)

for i in range(len(quantiles)):
    if i == 0:
        ax.plot(x_pred, quantiles[i], color='k', linestyle='dashed', alpha=.5, label='Percentiles (5, ' + r'$\bf{50}$' + ', 95)')
    elif i == 1:
        ax.plot(x_pred, quantiles[i], color='k', linewidth=2, linestyle='dashed', alpha=1)
    else:
        ax.plot(x_pred, quantiles[i], color='k', linestyle='dashed', alpha=.5)

ax.fill_between([lower_x, upper_x], t.trf(0), t.trf(50), color='k', alpha=.1)
ax.hlines(t.trf(6.7), min(X), t.trf(60), color='k', linewidth=2, alpha=.1)
ax.set_xticks([t.trf(i) for i in [0, 2, 5, 10, 20, 35, 60]])
ax.set_xticklabels([0, 2, 5, 10, 20, 35, 60])
ax.set_yticks([t.trf(i) for i in [0, 2, 5, 10, 20, 35, 60]])
ax.set_yticklabels([0, 2, 5, 10, 20, 35, 60], fontsize=12)
ax.set_xlabel('Panel-Derived TMB', fontsize=16)
ax.set_ylabel('Exomic TMB', fontsize=16)
ax.tick_params(length=3, width=1)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_position(['outward', 5])
ax.spines['left'].set_bounds(t.trf(0), t.trf(60))
ax.spines['bottom'].set_bounds(t.trf(0), t.trf(60))
ax.set_ylim(t.trf(0), t.trf(60))
ax.set_xlim(-.01, t.trf(60))
cax = ax.inset_axes([.02, 0.92, 0.2, 0.02])
cbar = fig.colorbar(pcm, ax=ax, cax=cax, ticks=[0, 1], orientation='horizontal')
cbar.ax.set_title('Probability\nDensity', y=1, pad=-15, x=1.4, fontsize=10)
plt.legend(frameon=False, loc=(.01, .95), fontsize=10)
plt.savefig(cwd / 'figures' / 'figure_3' / 'gls_fit.png', dpi=600)


low_group = t.trf(test_df['panel_tmb'].values) < upper_x
high_group = t.trf(test_df['panel_tmb'].values) > upper_x

times = test_df['OS.time'].values.astype(np.int16)
events = test_df['OS'].values.astype(np.int16)

p_value = logrank_test(times[low_group], times[high_group], event_observed_A=events[low_group], event_observed_B=events[high_group]).p_value

fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(bottom=.099)
fig.subplots_adjust(top=.98)
fig.subplots_adjust(left=.099)
fig.subplots_adjust(right=.969)

survtimes = list(zip(times[low_group], events[low_group]))
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

survtimes = list(zip(times[high_group], events[high_group]))
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
          ('  <95% Model Probability \n                  N=' + str(sum(low_group)),
           '  >95% Model Probability \n                  N=' + str(sum(high_group))),
          loc=(.65, .85),
          frameon=False,
          borderpad=0,
          handletextpad=.5,
          labelspacing=1,
          fontsize=10,
          ncol=1)
ax.text(.0, .97, 'Logrank p-value=%.1E' % p_value, transform=ax.transAxes)

plt.ylim(0, 105)
plt.xlim(0, max(times))

plt.savefig(cwd / 'figures' / 'figure_3' / 'gls_kaplan.pdf')


