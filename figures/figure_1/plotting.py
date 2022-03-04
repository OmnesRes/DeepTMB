import numpy as np
import pylab as plt
import pickle
import pandas as pd
import tensorflow_probability as tfp
import tensorflow as tf

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

x_pred, y_pred, X, Y, Y_pred = pickle.load(open(cwd / 'figures' / 'figure_1' / 'fit.pkl', 'rb'))
y_preds = tfp.distributions.LogNormal(loc=y_pred[:, 0], scale=np.exp(y_pred[:, 1]))
Y_preds = tfp.distributions.LogNormal(loc=Y_pred[:, 0], scale=np.exp(Y_pred[:, 1]))
lower_estimates = Y_preds.quantile(.16).numpy()
upper_estimates = Y_preds.quantile(.84).numpy()

##check percent of data within estimates
sum((Y[:, 0] >= lower_estimates) & (Y[:, 0] <= upper_estimates)) / len(Y)

##input output plot
fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(top=1,
bottom=0.068,
left=0.084,
right=1.0,
hspace=0.2,
wspace=0.2)
ax.scatter(np.log(X + 1), np.log(Y + 1), s=5, edgecolor='none', alpha=.15)
ax.plot(np.log(x_pred + 1), np.log(y_preds.mean() + 1), color='k', linestyle='-', alpha=.5)
ax.plot(np.log(x_pred + 1), np.log(y_preds.quantile(.84) + 1), color='k', linestyle='dashed', alpha=.3)
ax.plot(np.log(x_pred + 1), np.log(y_preds.quantile(.16) + 1), color='k', linestyle='dashed', alpha=.3)
ax.set_xticks([np.log(i+1) for i in range(81)])
ax.set_xticklabels([])
ax.set_yticks([np.log(i + 1) for i in [0, 1, 2, 3, 5, 10, 20, 40]])
ax.set_yticklabels(['0', '1', '2', '3', '5', '10', '20', '40'], fontsize=12)
ax.set_xlabel('Input Values', fontsize=16)
ax.set_ylabel('TMB', fontsize=16)
ax.tick_params(length=3, width=1)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_position(['outward', -5])
ax.spines['left'].set_position(['outward', -5])
ax.spines['left'].set_bounds(np.log(1 + 0), np.log(1 + 40))
ax.spines['bottom'].set_bounds(np.log(1 + 0), np.log(1 + 80))
plt.savefig(cwd / 'figures' / 'figure_1' / 'input_output.png', dpi=600)


##pred true plot
fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(top=1,
bottom=0.07,
left=0.096,
right=1.0,
hspace=0.2,
wspace=0.2)
ax.scatter(np.log(Y_preds.mean() + 1), np.log(Y + 1), s=5, edgecolor='none', alpha=.15)
ax.plot(np.log(y_preds.mean() + 1), np.log(y_preds.mean() + 1), color='k', linestyle='-', alpha=.5)
ax.plot(np.log(y_preds.mean() + 1), np.log(y_preds.quantile(.84) + 1), color='k', linestyle='dashed', alpha=.3)
ax.plot(np.log(y_preds.mean() + 1), np.log(y_preds.quantile(.16) + 1), color='k', linestyle='dashed', alpha=.3)
ax.set_xticks([np.log(i+1) for i in [0, 1, 2, 3, 5, 10, 20, 40]])
ax.set_xticklabels(['0', '1', '2', '3', '5', '10', '20', '40'], fontsize=12)
ax.set_yticks([np.log(i + 1) for i in [0, 1, 2, 3, 5, 10, 20, 40]])
ax.set_yticklabels(['0', '1', '2', '3', '5', '10', '20', '40'], fontsize=12)
ax.set_xlabel('Estimated TMB', fontsize=16)
ax.set_ylabel('TMB', fontsize=16)
ax.tick_params(length=3, width=1)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_bounds(np.log(1 + 0), np.log(1 + 40))
ax.spines['bottom'].set_bounds(np.log(1 + 0), np.log(1 + 40))
ax.spines['bottom'].set_position(['outward', -9])
plt.savefig(cwd / 'figures' / 'figure_1' / 'pred_true.png', dpi=600)

