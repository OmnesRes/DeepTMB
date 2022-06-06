import numpy as np
import tensorflow as tf
from model.model import Encoders, NN
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

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[-3], True)
tf.config.experimental.set_visible_devices(physical_devices[-3], 'GPU')

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

data = pickle.load(open(cwd / 'figures' / 'figure_3' / 'data' / 'data.pkl', 'rb'))
sample_table = pickle.load(open(cwd / 'files' / 'tcga_public_sample_table.pkl', 'rb'))
germline_samples = pickle.load(open(cwd / 'files' / 'germline' / 'data' / 'germline_samples.pkl', 'rb'))

[germline_samples.pop(i) for i in list(germline_samples.keys()) if germline_samples[i] < 400]
[data.pop(i) for i in list(data.keys()) if not data[i]]
[data.pop(i) for i in list(data.keys()) if i[:12] not in germline_samples]
nci_dict = {i: j for i, j in zip(sample_table['Tumor_Sample_Barcode'].values, sample_table['NCIt_tmb_label'].values) if j}
[data.pop(i) for i in list(data.keys()) if i not in nci_dict]

cutoff = np.percentile([i[0][2] / (i[1] / 1e6) for i in data.values()], 98)
values = [i for i in data.values() if (i[0][2] / (i[1] / 1e6)) < cutoff]
nci = np.array([nci_dict[i] for i in data if (data[i][0][2] / (data[i][1] / 1e6)) < cutoff])

X = np.array([i[0][2] / (i[1] / 1e6) for i in values])
Y = np.array([i[2] / (i[3] / 1e6) for i in values])
y_weights = np.ones_like(Y)


t = utils.LogTransform(bias=4, min_x=0)
X = t.trf(X[:, np.newaxis])
Y = t.trf(Y)
X_loader = utils.Map.PassThrough(X)
Y_loader = utils.Map.PassThrough(Y)
Y_loader_fcn = utils.Map.PassThrough(Y[:, np.newaxis])
W_loader = utils.Map.PassThrough(y_weights)


mixture_count_encoder = Encoders.Encoder(shape=(1,), layers=(128,))
mixture_net = NN(encoders=[mixture_count_encoder.model], layers=(64, 32), mode='mixture')

mixture_net.model.compile(loss=utils.log_prob_loss,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

fcn_count_encoder = Encoders.Encoder(shape=(1,), layers=(128,))
fcn_net = NN(encoders=[fcn_count_encoder.model], layers=(64, 32), mode='fcn')
fcn_net.model.compile(loss='mse',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.001, patience=30, mode='min', restore_best_weights=True)]

idx_train = np.arange(len(X))
ds_train = tf.data.Dataset.from_tensor_slices((idx_train, ))
ds_train = ds_train.shuffle(buffer_size=len(idx_train), reshuffle_each_iteration=True).repeat().batch(batch_size=int(len(idx_train) * .75), drop_remainder=True)
ds_train_mixture = ds_train.map(lambda x: ((
                                    X_loader(x),
                                    ),
                                   (Y_loader(x),
                                    ),
                                    W_loader(x)
                                   )
                        ).map(utils.rescale_batch_weights)

mixture_net.model.fit(ds_train_mixture,
              steps_per_epoch=10,
              epochs=10000,
              callbacks=callbacks
              )

ds_train_fcn = ds_train.map(lambda x: ((
                                    X_loader(x),
                                    ),
                                   (Y_loader_fcn(x),
                                    ),
                                    W_loader(x)
                                   )
                        ).map(utils.rescale_batch_weights)

fcn_net.model.fit(ds_train_fcn,
              steps_per_epoch=10,
              epochs=10000,
              callbacks=callbacks
              )

x_pred = np.linspace(np.min(X), np.max(X), 200)
y_pred = np.linspace(np.min(Y), np.max(Y + .5), 1000)
Z = mixture_net.model(x_pred).log_prob(y_pred[:, np.newaxis]).numpy()
zz = np.exp(Z - Z.max(axis=0, keepdims=True))
X_pred, Y_pred = np.meshgrid(x_pred, y_pred)

mixture_means = mixture_net.model(x_pred).mean()
fcn_means = fcn_net.model.predict(x_pred)
quantiles = [y_pred[np.argmin(np.diff((np.exp(mixture_net.model(x_pred).log_cdf(y_pred[:, np.newaxis]).numpy()) < 0.05).astype(int), axis=0), axis=0)],
             y_pred[np.argmin(np.diff((np.exp(mixture_net.model(x_pred).log_cdf(y_pred[:, np.newaxis]).numpy()) < 0.25).astype(int), axis=0), axis=0)],
             y_pred[np.argmin(np.diff((np.exp(mixture_net.model(x_pred).log_cdf(y_pred[:, np.newaxis]).numpy()) < 0.5).astype(int), axis=0), axis=0)],
             y_pred[np.argmin(np.diff((np.exp(mixture_net.model(x_pred).log_cdf(y_pred[:, np.newaxis]).numpy()) < 0.75).astype(int), axis=0), axis=0)],
            y_pred[np.argmin(np.diff((np.exp(mixture_net.model(x_pred).log_cdf(y_pred[:, np.newaxis]).numpy()) < 0.95).astype(int), axis=0), axis=0)]]
medians = y_pred[np.argmin(np.diff((np.exp(mixture_net.model(X).log_cdf(y_pred[:, np.newaxis]).numpy()) < 0.5).astype(int), axis=0), axis=0)]


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
ax.scatter(X, Y, s=5, edgecolor='none', alpha=.3)
ax.plot(x_pred, mixture_means, color='k', linestyle='-', alpha=1, label='GMM means')
ax.scatter(x_pred[::6], fcn_means[:, 0][::6], color='k', marker='o', alpha=1, s=10, label='FCN predictions')
for i in range(len(quantiles)):
    if i==0:
        ax.plot(x_pred, quantiles[i], color='k', linestyle='dashed', alpha=.5, label='Percentiles (5, 25, 50, 75, 95)')
    else:
        ax.plot(x_pred, quantiles[i], color='k', linestyle='dashed', alpha=.5)

ax.hlines(-.05, t.trf(3), t.trf(5), color='r', linewidth=1)
ax.hlines(-.05, t.trf(7), t.trf(10), color='r', linewidth=1)
ax.hlines(-.05, t.trf(15), t.trf(20), color='r', linewidth=1)
ax.hlines(-.05, t.trf(25), t.trf(35), color='r', linewidth=1)
ax.set_xticks([t.trf(i) for i in [0, 2, 5, 10, 20, 35, 60]])
ax.set_xticklabels([0, 2, 5, 10, 20, 35, 60])
ax.set_yticks([t.trf(i) for i in [0, 2, 5, 10, 20, 35, 60]])
ax.set_yticklabels([0, 2, 5, 10, 20, 35, 60], fontsize=12)
ax.set_xlabel('Panel-Derived Input Values', fontsize=16)
ax.set_ylabel('Exomic TMB', fontsize=16)
ax.tick_params(length=3, width=1)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_position(['outward', 5])
ax.spines['left'].set_bounds(t.trf(0), t.trf(60))
ax.spines['bottom'].set_bounds(t.trf(0), t.trf(60))
ax.set_ylim(-.1, t.trf(60))
ax.set_xlim(-.01, t.trf(60))
cax = ax.inset_axes([.025, 0.82, 0.2, 0.02])
cbar = fig.colorbar(pcm, ax=ax, cax=cax, ticks=[0, 1], orientation='horizontal')
cbar.ax.set_title('Probability\nDensity', y=1, pad=-15, x=1.4, fontsize=10)
plt.legend(frameon=False, loc=(.01, .86), fontsize=10)
plt.savefig(cwd / 'figures' / 'figure_1' / 'results' / 'germline_gmm_input_output.png', dpi=600)


##distributions
fig = plt.figure()
all_ax = fig.add_subplot(111)
all_ax.spines['right'].set_visible(False)
all_ax.spines['top'].set_visible(False)
all_ax.spines['bottom'].set_visible(False)
all_ax.spines['left'].set_visible(False)
all_ax.set_yticks([])
all_ax.set_xticks([])
all_ax.set_title('Panel-Derived Input Values', y=0, fontsize=16)
gs = fig.add_gridspec(1, 4, bottom=.07)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1], sharey=ax1, sharex=ax1)
ax3 = fig.add_subplot(gs[0, 2], sharey=ax1, sharex=ax1)
ax4 = fig.add_subplot(gs[0, 3], sharey=ax1, sharex=ax1)

fig.subplots_adjust(top=1,
bottom=0,
left=0.047,
right=1,
hspace=0.2,
wspace=0.136)

##round data at .1 and bin at .1
bin = .1
x_mask = [(i > 3) and (i < 5) for i in t.inv(X[:, 0])]
temp_y = Y[x_mask]
temp_x = X[x_mask]
counts = dict(zip(*np.unique(np.around(temp_y, 1), return_counts=True)))
counts.update({round(i * bin, 1): counts.get(round(i * bin, 1), 0) for i in range(0, int(t.trf(61) / bin))})
values = [sum([counts.get((i / 10) + j / 10, 0) for j in range(int(bin * 10))]) for i in range(int(min(counts.keys()) * 10), int(max(counts.keys()) * 10) + 1, int(bin * 10))]

y_pred = np.linspace(min(counts.keys()), max(list(counts.keys())), 200)
Z = mixture_net.model(np.array([t.trf(4)])[:, np.newaxis]).log_prob(y_pred[:, np.newaxis]).numpy()
zz = np.exp(Z - Z.max(axis=0, keepdims=True))
zz = zz / np.sum(zz, axis=0) * (200 / len([i / 10 for i in range(int(min(counts.keys()) * 10), int(max(counts.keys()) * 10) + 1, int(bin * 10))]))

ax1.barh([i / 10 for i in range(int(min(counts.keys()) * 10), int(max(counts.keys()) * 10) + 1, int(bin * 10))], np.array(values) / sum(values), bin, alpha=.3)
ax1.plot(zz, y_pred, color='#00ff00')
ax1.set_xlabel('3-5', labelpad=-12)

x_mask = [(i > 7) and (i < 10) for i in t.inv(X[:, 0])]
temp_y = Y[x_mask]
temp_x = X[x_mask]
counts = dict(zip(*np.unique(np.around(temp_y, 1), return_counts=True)))
counts.update({round(i * bin, 1): counts.get(round(i * bin, 1), 0) for i in range(0, int(t.trf(61) / bin))})
values = [sum([counts.get((i / 10) + j / 10, 0) for j in range(int(bin * 10))]) for i in range(int(min(counts.keys()) * 10), int(max(counts.keys()) * 10) + 1, int(bin * 10))]

y_pred = np.linspace(min(counts.keys()), max(list(counts.keys())), 200)
Z = mixture_net.model(np.array([t.trf(8.5)])[:, np.newaxis]).log_prob(y_pred[:, np.newaxis]).numpy()
zz = np.exp(Z - Z.max(axis=0, keepdims=True))
zz = zz / np.sum(zz, axis=0) * (200 / len([i / 10 for i in range(int(min(counts.keys()) * 10), int(max(counts.keys()) * 10) + 1, int(bin * 10))]))

ax2.barh([i / 10 for i in range(int(min(counts.keys()) * 10), int(max(counts.keys()) * 10) + 1, int(bin * 10))], np.array(values) / sum(values), bin, alpha=.3)
ax2.plot(zz, y_pred, color='#00ff00')
ax2.set_xlabel('7-10', labelpad=-14)

x_mask = [(i > 15) and (i < 20) for i in t.inv(X[:, 0])]
temp_y = Y[x_mask]
temp_x = X[x_mask]
counts = dict(zip(*np.unique(np.around(temp_y, 1), return_counts=True)))
counts.update({round(i * bin, 1): counts.get(round(i * bin, 1), 0) for i in range(0, int(t.trf(61) / bin))})
values = [sum([counts.get((i / 10) + j / 10, 0) for j in range(int(bin * 10))]) for i in range(int(min(counts.keys()) * 10), int(max(counts.keys()) * 10) + 1, int(bin * 10))]

y_pred = np.linspace(min(counts.keys()), max(list(counts.keys())), 200)
Z = mixture_net.model(np.array([t.trf(17.5)])[:, np.newaxis]).log_prob(y_pred[:, np.newaxis]).numpy()
zz = np.exp(Z - Z.max(axis=0, keepdims=True))
zz = zz / np.sum(zz, axis=0) * (200 / len([i / 10 for i in range(int(min(counts.keys()) * 10), int(max(counts.keys()) * 10) + 1, int(bin * 10))]))

ax3.barh([i / 10 for i in range(int(min(counts.keys()) * 10), int(max(counts.keys()) * 10) + 1, int(bin * 10))], np.array(values) / sum(values), bin, alpha=.3)
ax3.plot(zz, y_pred, color='#00ff00')
ax3.set_xlabel('15-20', labelpad=-14)

x_mask = [(i > 25) and (i < 35) for i in t.inv(X[:, 0])]
temp_y = Y[x_mask]
temp_x = X[x_mask]
counts = dict(zip(*np.unique(np.around(temp_y, 1), return_counts=True)))
counts.update({round(i * bin, 1): counts.get(round(i * bin, 1), 0) for i in range(0, int(t.trf(61) / bin))})
values = [sum([counts.get((i / 10) + j / 10, 0) for j in range(int(bin * 10))]) for i in range(int(min(counts.keys()) * 10), int(max(counts.keys()) * 10) + 1, int(bin * 10))]

y_pred = np.linspace(min(counts.keys()), max(list(counts.keys())), 200)
Z = mixture_net.model(np.array([t.trf(30)])[:, np.newaxis]).log_prob(y_pred[:, np.newaxis]).numpy()
zz = np.exp(Z - Z.max(axis=0, keepdims=True))
zz = zz / np.sum(zz, axis=0) * (200 / len([i / 10 for i in range(int(min(counts.keys()) * 10), int(max(counts.keys()) * 10) + 1, int(bin * 10))]))

ax4.barh([i / 10 for i in range(int(min(counts.keys()) * 10), int(max(counts.keys()) * 10) + 1, int(bin * 10))], np.array(values) / sum(values), bin, alpha=.3)
ax4.plot(zz, y_pred, color='#00ff00')
ax4.set_xlabel('25-35', labelpad=-14)

for ax in [ax1, ax2, ax3, ax4]:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([t.trf(i) for i in [0, 2, 5, 10, 20, 35, 60]])
    ax.set_yticklabels([0, 2, 5, 10, 20, 35, 60])
    ax.spines['left'].set_bounds(t.trf(0), t.trf(60))

plt.savefig(cwd / 'figures' / 'figure_1' / 'results' / 'germline_gmm_distributions.png', dpi=600)

##residuals plot

fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(top=.98,
bottom=0.12,
left=0.11,
right=.98,
hspace=0.2,
wspace=0.2)
ax.scatter(X, Y - medians, s=5, edgecolor='none', alpha=.3)
ax.axhline(y=0, c='k', linestyle='-', linewidth=2, alpha=0.5)
for i in range(5):
    if i != 2:
        ax.plot(x_pred[:190], (quantiles[i] - quantiles[2])[:190], color='k', linestyle='dashed', alpha=.5)
ax.set_xticks([t.trf(i) for i in [0, 2, 5, 10, 20, 35, 60]])
ax.set_xticklabels([0, 2, 5, 10, 20, 35, 60])
ax.set_yticks([-2, -1.5, -1, -.5, 0, .5, 1, 1.5, 2])
ax.set_yticklabels([-2, -1.5, -1, -.5, 0, .5, 1, 1.5, 2], fontsize=12)
ax.set_xlabel('Panel-Derived Input Values', fontsize=16)
ax.set_ylabel('Residuals', fontsize=16)
ax.tick_params(length=3, width=1)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_position(['outward', 5])
ax.spines['left'].set_position(['outward', 5])
ax.spines['left'].set_bounds(-2, 2)
ax.spines['bottom'].set_bounds(t.trf(0), t.trf(60))
ax.set_ylim(-2, 2)
ax.set_xlim(-.01, t.trf(60))
plt.savefig(cwd / 'figures' / 'figure_1' / 'results' / 'germline_gmm_residuals.png', dpi=600)

