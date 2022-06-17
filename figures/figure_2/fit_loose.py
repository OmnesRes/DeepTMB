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
tf.config.experimental.set_memory_growth(physical_devices[-4], True)
tf.config.experimental.set_visible_devices(physical_devices[-4], 'GPU')

data = pickle.load(open(cwd / 'tables' / 'table_1' / 'DUKE-F1-DX1' / 'tumor_only_loose' / 'data' / 'data.pkl', 'rb'))
germline_samples = pickle.load(open(cwd / 'files' / 'germline' / 'data' / 'germline_samples.pkl', 'rb'))
ancestry = pickle.load(open(cwd / 'files' / 'ethnicity.pkl', 'rb'))

[data.pop(i) for i in list(data.keys()) if not data[i]]
[germline_samples.pop(i) for i in list(germline_samples.keys()) if germline_samples[i] < 400]
[data.pop(i) for i in list(data.keys()) if i[:12] not in germline_samples]

cutoff = np.percentile([i[-1] / (i[1] / 1e6) for i in data.values()], 98)
values = [i for i in data.values() if (i[-1] / (i[1] / 1e6)) < cutoff]

anc = np.array([ancestry.get(i[:12], 'OA') for i in data if (data[i][-1] / (data[i][1] / 1e6)) < cutoff])
anc_encoding = {'AA': 1, 'EA': 2, 'EAA': 3, 'NA': 4, 'OA': 0}
anc = np.array([anc_encoding[i] for i in anc])

X = np.array([i[-1] / (i[1] / 1e6) for i in values])
Y = np.array([i[2] / (i[3] / 1e6) for i in values])

anc_counts = dict(zip(*np.unique(anc, return_counts=True)))
y_weights = np.array([1 / anc_counts[_] for _ in anc])
y_weights /= np.sum(y_weights)

t = utils.LogTransform(bias=4, min_x=0)
X = t.trf(X[:, np.newaxis])
Y = t.trf(Y)
anc_loader = utils.Map.PassThrough(anc)
X_loader = utils.Map.PassThrough(X)
Y_loader = utils.Map.PassThrough(Y)
W_loader = utils.Map.PassThrough(y_weights)

count_encoder = Encoders.Encoder(shape=(1,), layers=(128,))
anc_encoder = Encoders.Embedder(shape=(), layers=(128,), dim=4)
net = NN(encoders=[count_encoder.model, anc_encoder.model], layers=(64, 32), mode='mixture')

net.model.compile(loss=utils.log_prob_loss,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.001, patience=30, mode='min', restore_best_weights=True)]

idx_train = np.arange(len(X))
ds_train = tf.data.Dataset.from_tensor_slices((idx_train, ))
ds_train = ds_train.shuffle(buffer_size=len(idx_train), reshuffle_each_iteration=True).repeat().batch(batch_size=int(len(idx_train) * .75), drop_remainder=True)
ds_train_mixture = ds_train.map(lambda x: ((
                                    X_loader(x),
                                    anc_loader(x),
                                    ),
                                   (Y_loader(x),
                                    ),
                                    W_loader(x)
                                   )
                        ).map(utils.rescale_batch_weights)

net.model.fit(ds_train_mixture,
              steps_per_epoch=10,
              epochs=10000,
              callbacks=callbacks
              )


x_pred = np.linspace(np.min(X), np.max(X), 200)
y_pred = np.linspace(np.min(Y), np.max(Y + .5), 1000)

medians = []
quantiles = []
for i in range(1, 5):
    mask = anc == i
    medians.append(y_pred[np.argmin(np.diff((np.exp(net.model((X[mask], anc[mask])).log_cdf(y_pred[:, np.newaxis]).numpy()) < 0.5).astype(int), axis=0), axis=0)])
    quantiles.append([y_pred[np.argmin(np.diff((np.exp(net.model((x_pred, np.repeat(i, 200))).log_cdf(y_pred[:, np.newaxis]).numpy()) < 0.05).astype(int), axis=0), axis=0)],
                 y_pred[np.argmin(np.diff((np.exp(net.model((x_pred, np.repeat(i, 200))).log_cdf(y_pred[:, np.newaxis]).numpy()) < 0.25).astype(int), axis=0), axis=0)],
                 y_pred[np.argmin(np.diff((np.exp(net.model((x_pred, np.repeat(i, 200))).log_cdf(y_pred[:, np.newaxis]).numpy()) < 0.5).astype(int), axis=0), axis=0)],
                 y_pred[np.argmin(np.diff((np.exp(net.model((x_pred, np.repeat(i, 200))).log_cdf(y_pred[:, np.newaxis]).numpy()) < 0.75).astype(int), axis=0), axis=0)],
                 y_pred[np.argmin(np.diff((np.exp(net.model((x_pred, np.repeat(i, 200))).log_cdf(y_pred[:, np.newaxis]).numpy()) < 0.95).astype(int), axis=0), axis=0)]])

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
        ax.plot(x_pred, quantiles[anc_encoding[label] - 1][2], linestyle='-', alpha=.7, label=label_names[label], color=colors[anc_encoding[label] - 1])
        # ax.vlines(x_pred[np.argmin(np.abs(t.trf(10) - quantiles[anc_encoding[label] - 1][2]))], -.1, t.trf(10), linestyles='dotted', colors=colors[anc_encoding[label] - 1], alpha=1)
ax.hlines(t.trf(10), -.1, t.trf(40), color='k', zorder=-1000, alpha=.3)
ax.vlines(t.trf(27), -.3, t.trf(40), linestyles='dotted', colors='k', alpha=.3)
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