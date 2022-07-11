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
tf.config.experimental.set_memory_growth(physical_devices[-2], True)
tf.config.experimental.set_visible_devices(physical_devices[-2], 'GPU')

data = pickle.load(open(cwd / 'tables' / 'table_1' / 'DUKE-F1-DX1' / 'tumor_normal' / 'data' / 'data.pkl', 'rb'))

[data.pop(i) for i in list(data.keys()) if not data[i]]

non_syn = ['Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins', 'In_Frame_Del', 'In_Frame_Ins', 'Nonstop_Mutation']
non_syn_data = {i: sum([data[i][5].to_dict()[j] for j in data[i][5].index if j in non_syn]) for i in data}
cutoff = np.percentile(list(non_syn_data.values()), 98)
mask = list(non_syn_data.values()) < cutoff
X = np.array([i[0] / (i[1] / 1e6) for i in data.values()])
Y = np.array([i[2] / (i[3] / 1e6) for i in data.values()])
y_weights = np.ones_like(Y)

t = utils.LogTransform(bias=4, min_x=0)
X = t.trf(X[mask, np.newaxis])
Y = t.trf(Y[mask])
X_loader = utils.Map.PassThrough(X)
Y_loader = utils.Map.PassThrough(Y)
W_loader = utils.Map.PassThrough(y_weights[mask])


count_encoder = Encoders.Encoder(shape=(1,), layers=(128,))
net = NN(encoders=[count_encoder.model], layers=(64, 32), mode='mixture')

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


x_pred = np.linspace(np.min(X), np.max(X), 200)
y_pred = np.linspace(0, np.max(Y + .5), 1000)
Z = net.model(x_pred).log_prob(y_pred[:, np.newaxis]).numpy()
zz = np.exp(Z - Z.max(axis=0, keepdims=True))
X_pred, Y_pred = np.meshgrid(x_pred, y_pred)

quantiles = [y_pred[np.argmin(np.diff((np.exp(net.model(x_pred).log_cdf(y_pred[:, np.newaxis]).numpy()) < 0.05).astype(int), axis=0), axis=0)],
             y_pred[np.argmin(np.diff((np.exp(net.model(x_pred).log_cdf(y_pred[:, np.newaxis]).numpy()) < 0.25).astype(int), axis=0), axis=0)],
             y_pred[np.argmin(np.diff((np.exp(net.model(x_pred).log_cdf(y_pred[:, np.newaxis]).numpy()) < 0.5).astype(int), axis=0), axis=0)],
             y_pred[np.argmin(np.diff((np.exp(net.model(x_pred).log_cdf(y_pred[:, np.newaxis]).numpy()) < 0.75).astype(int), axis=0), axis=0)],
             y_pred[np.argmin(np.diff((np.exp(net.model(x_pred).log_cdf(y_pred[:, np.newaxis]).numpy()) < 0.95).astype(int), axis=0), axis=0)]]

medians = y_pred[np.argmin(np.diff((np.exp(net.model(X).log_cdf(y_pred[:, np.newaxis]).numpy()) < 0.5).astype(int), axis=0), axis=0)]

##pred true plot
fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(top=.98,
bottom=0.11,
left=0.096,
right=.985,
hspace=0.2,
wspace=0.2)
ax.scatter(medians, Y, s=5, edgecolor='none', alpha=.3)
for i in range(len(quantiles)):
    if i == 0:
        ax.plot(quantiles[2], quantiles[i], color='k', linestyle='dashed', alpha=.5, label='Percentiles (5, 25, ' + r'$\bf{50}$' + ', 75, 95)')
    elif i == 2:
        ax.plot(quantiles[2], quantiles[i], color='k', linewidth=2, linestyle='dashed', alpha=1)
    else:
        ax.plot(quantiles[2], quantiles[i], color='k', linestyle='dashed', alpha=.5)

ax.set_xticks([t.trf(i) for i in [0, 2, 5, 10, 20, 35]])
ax.set_xticklabels([0, 2, 5, 10, 20, 35])
ax.set_yticks([t.trf(i) for i in [0, 1, 2, 3, 5, 10, 20, 35, 60]])
ax.set_yticklabels(['0', '1', '2', '3', '5', '10', '20', '35', '60'], fontsize=12)
ax.set_xlabel('Estimated TMB', fontsize=16)
ax.set_ylabel('Exomic TMB', fontsize=16)
ax.tick_params(length=3, width=1)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_bounds(t.trf(0), t.trf(60))
ax.spines['bottom'].set_bounds(t.trf(0), t.trf(35))
ax.spines['bottom'].set_position(['outward', 5])
ax.set_ylim(t.trf(0), t.trf(60))
ax.set_xlim(t.trf(-.01), t.trf(35))
plt.legend(frameon=False, loc=(.01, .95), fontsize=10)
plt.savefig(cwd / 'figures' / 'supplemental_figures' / 'figure_1' / 'results' / 'normal.png', dpi=600)


data = pickle.load(open(cwd / 'tables' / 'table_1' / 'DUKE-F1-DX1' / 'tumor_only_strict' / 'data' / 'data.pkl', 'rb'))
germline_samples = pickle.load(open(cwd / 'files' / 'germline' / 'data' / 'germline_samples.pkl', 'rb'))

[data.pop(i) for i in list(data.keys()) if not data[i]]
[germline_samples.pop(i) for i in list(germline_samples.keys()) if germline_samples[i] < 400]
[data.pop(i) for i in list(data.keys()) if i[:12] not in germline_samples]

cutoff = np.percentile([i[-1] / (i[1] / 1e6) for i in data.values()], 98)
mask = [(i[-1] / (i[1] / 1e6)) < cutoff for i in data.values()]

X = np.array([i[0] / (i[1] / 1e6) for i in data.values()])
Y = np.array([i[2] / (i[3] / 1e6) for i in data.values()])

t = utils.LogTransform(bias=4, min_x=0)
X = t.trf(X[mask, np.newaxis])
Y = t.trf(Y[mask])
medians = y_pred[np.argmin(np.diff((np.exp(net.model(X).log_cdf(y_pred[:, np.newaxis]).numpy()) < 0.5).astype(int), axis=0), axis=0)]

##pred true plot
fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(top=.98,
bottom=0.11,
left=0.096,
right=.985,
hspace=0.2,
wspace=0.2)
ax.scatter(medians, Y, s=5, edgecolor='none', alpha=.3)
for i in range(len(quantiles)):
    if i == 0:
        ax.plot(quantiles[2], quantiles[i], color='k', linestyle='dashed', alpha=.5, label='Percentiles (5, 25, ' + r'$\bf{50}$' + ', 75, 95)')
    elif i == 2:
        ax.plot(quantiles[2], quantiles[i], color='k', linewidth=2, linestyle='dashed', alpha=1)
    else:
        ax.plot(quantiles[2], quantiles[i], color='k', linestyle='dashed', alpha=.5)

ax.set_xticks([t.trf(i) for i in [0, 2, 5, 10, 20, 35]])
ax.set_xticklabels([0, 2, 5, 10, 20, 35])
ax.set_yticks([t.trf(i) for i in [0, 1, 2, 3, 5, 10, 20, 35, 60]])
ax.set_yticklabels(['0', '1', '2', '3', '5', '10', '20', '35', '60'], fontsize=12)
ax.set_xlabel('Estimated TMB', fontsize=16)
ax.set_ylabel('Exomic TMB', fontsize=16)
ax.tick_params(length=3, width=1)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_bounds(t.trf(0), t.trf(60))
ax.spines['bottom'].set_bounds(t.trf(0), t.trf(35))
ax.spines['bottom'].set_position(['outward', 5])
ax.set_ylim(t.trf(0), t.trf(60))
ax.set_xlim(t.trf(-.01), t.trf(35))
plt.legend(frameon=False, loc=(.01, .95), fontsize=10)
plt.savefig(cwd / 'figures' / 'supplemental_figures' / 'figure_1' / 'results' / 'strict.png', dpi=600)


data = pickle.load(open(cwd / 'tables' / 'table_1' / 'DUKE-F1-DX1' / 'tumor_only_loose' / 'data' / 'data.pkl', 'rb'))
germline_samples = pickle.load(open(cwd / 'files' / 'germline' / 'data' / 'germline_samples.pkl', 'rb'))

[data.pop(i) for i in list(data.keys()) if not data[i]]
[germline_samples.pop(i) for i in list(germline_samples.keys()) if germline_samples[i] < 400]
[data.pop(i) for i in list(data.keys()) if i[:12] not in germline_samples]

cutoff = np.percentile([i[-1] / (i[1] / 1e6) for i in data.values()], 98)
mask = [(i[-1] / (i[1] / 1e6)) < cutoff for i in data.values()]

X = np.array([i[0] / (i[1] / 1e6) for i in data.values()])
Y = np.array([i[2] / (i[3] / 1e6) for i in data.values()])

t = utils.LogTransform(bias=4, min_x=0)
X = t.trf(X[mask, np.newaxis])
Y = t.trf(Y[mask])
medians = y_pred[np.argmin(np.diff((np.exp(net.model(X).log_cdf(y_pred[:, np.newaxis]).numpy()) < 0.5).astype(int), axis=0), axis=0)]


##pred true plot
fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(top=.98,
bottom=0.11,
left=0.096,
right=.985,
hspace=0.2,
wspace=0.2)
ax.scatter(medians, Y, s=5, edgecolor='none', alpha=.3)
for i in range(len(quantiles)):
    if i == 0:
        ax.plot(quantiles[2], quantiles[i], color='k', linestyle='dashed', alpha=.5, label='Percentiles (5, 25, ' + r'$\bf{50}$' + ', 75, 95)')
    elif i == 2:
        ax.plot(quantiles[2], quantiles[i], color='k', linewidth=2, linestyle='dashed', alpha=1)
    else:
        ax.plot(quantiles[2], quantiles[i], color='k', linestyle='dashed', alpha=.5)

ax.set_xticks([t.trf(i) for i in [0, 2, 5, 10, 20, 35]])
ax.set_xticklabels([0, 2, 5, 10, 20, 35])
ax.set_yticks([t.trf(i) for i in [0, 1, 2, 3, 5, 10, 20, 35, 60]])
ax.set_yticklabels(['0', '1', '2', '3', '5', '10', '20', '35', '60'], fontsize=12)
ax.set_xlabel('Estimated TMB', fontsize=16)
ax.set_ylabel('Exomic TMB', fontsize=16)
ax.tick_params(length=3, width=1)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_bounds(t.trf(0), t.trf(60))
ax.spines['bottom'].set_bounds(t.trf(0), t.trf(35))
ax.spines['bottom'].set_position(['outward', 5])
ax.set_ylim(t.trf(0), t.trf(60))
ax.set_xlim(t.trf(-.01), t.trf(35))
plt.legend(frameon=False, loc=(.01, .95), fontsize=10)
plt.savefig(cwd / 'figures' / 'supplemental_figures' / 'figure_1' / 'results' / 'loose.png', dpi=600)
