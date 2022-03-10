import numpy as np
import pylab as plt
import pickle
import seaborn as sns
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
tf.config.experimental.set_memory_growth(physical_devices[-1], True)
tf.config.experimental.set_visible_devices(physical_devices[-1], 'GPU')

x_pred, y_pred, X, Y, Y_pred = pickle.load(open(cwd / 'figures' / 'figure_2' / 'augmented_fit.pkl', 'rb'))
y_preds = tfp.distributions.LogNormal(loc=y_pred[:, 0], scale=np.exp(y_pred[:, 1]))
Y_preds = tfp.distributions.LogNormal(loc=Y_pred[:, 0], scale=np.exp(Y_pred[:, 1]))
print(round(np.sqrt(np.mean((Y_preds.mean() - np.mean(Y, axis=-1))**2)), 2))

mean_augmented_x_pred, mean_augmented_y_pred, X, Y, mean_augmented_Y_pred = pickle.load(open(cwd / 'figures' / 'figure_2' / 'augmented_fit_once.pkl', 'rb'))
mean_augmented_y_preds = tfp.distributions.LogNormal(loc=mean_augmented_y_pred[:, 0], scale=np.exp(mean_augmented_y_pred[:, 1]))
Y_preds = tfp.distributions.LogNormal(loc=mean_augmented_Y_pred[:, 0], scale=np.exp(mean_augmented_Y_pred[:, 1]))
print(round(np.sqrt(np.mean((Y_preds.mean() - np.mean(Y, axis=-1))**2)), 2))

unaugmented_x_pred, unaugmented_y_pred, unaugmented_X, unaugmented_Y, unaugmented_Y_pred = pickle.load(open(cwd / 'figures' / 'figure_2' / 'unaugmented_fit.pkl', 'rb'))
unaugmented_y_preds = tfp.distributions.LogNormal(loc=unaugmented_y_pred[:, 0], scale=np.exp(unaugmented_y_pred[:, 1]))
Y_preds = tfp.distributions.LogNormal(loc=unaugmented_Y_pred[:, 0], scale=np.exp(unaugmented_Y_pred[:, 1]))
print(round(np.sqrt(np.mean((Y_preds.mean() - unaugmented_Y[:,0])**2)), 2))


from matplotlib import cm
paired = [cm.get_cmap('Paired')(i) for i in range(12) if i not in [4, 5]]


##input output plot
fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(top=1,
bottom=0.068,
left=0.084,
right=1.0,
hspace=0.2,
wspace=0.2)
ax.scatter(np.log(np.concatenate([i[:5] for i in X]) + 1), np.log(np.concatenate([i[:5] for i in Y]) + 1), s=5, edgecolor='none', alpha=.1, color=paired[1])
ax.plot(np.log(unaugmented_x_pred + 1), np.log(unaugmented_y_preds.quantile(.84) + 1), color='k', linestyle='-', alpha=1, label='Unaugmented')
ax.plot(np.log(unaugmented_x_pred + 1), np.log(unaugmented_y_preds.quantile(.16) + 1), color='k', linestyle='-', alpha=1)
ax.plot(np.log(mean_augmented_x_pred + 1), np.log(mean_augmented_y_preds.quantile(.84) + 1), color='k', linestyle='dashed', alpha=1, label='Augmented')
ax.plot(np.log(mean_augmented_x_pred + 1), np.log(mean_augmented_y_preds.quantile(.16) + 1), color='k', linestyle='dashed', alpha=1)
ax.plot(np.log(x_pred + 1), np.log(y_preds.quantile(.84) + 1), color='k', linestyle='dashed', alpha=.5, label='Augmented per batch')
ax.plot(np.log(x_pred + 1), np.log(y_preds.quantile(.16) + 1), color='k', linestyle='dashed', alpha=.5)
ax.set_xticks([np.log(i+1) for i in range(109)])
ax.set_xticklabels([])
ax.set_yticks([np.log(i + 1) for i in [0, 1, 2, 3, 5, 10, 20, 35, 60]])
ax.set_yticklabels(['0', '1', '2', '3', '5', '10', '20', '35', '60'], fontsize=12)
ax.set_xlabel('Augmented Input Values', fontsize=16)
ax.set_ylabel('Augmented TMB', fontsize=16)
ax.tick_params(length=3, width=1)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_position(['outward', -5])
ax.spines['left'].set_position(['outward', -5])
ax.spines['left'].set_bounds(np.log(1 + 0), np.log(1 + 60))
ax.spines['bottom'].set_bounds(np.log(1 + 0), np.log(1 + 109))
ax.legend(frameon=False, loc=(.02, .75))
plt.savefig(cwd / 'figures' / 'figure_2' / 'input_output.png', dpi=600)


##augmentation versus TMB plot

fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(top=1,
bottom=0.084,
left=0.084,
right=1.0,
hspace=0.2,
wspace=0.2)
sns.regplot(np.log(unaugmented_Y + 1),
            np.mean(Y, axis=-1, keepdims=True) / unaugmented_Y,
            ax=ax,
            truncate=True,
            scatter_kws={'s': 15,
                         'edgecolor': 'none',
                         'alpha': .2},
            line_kws={'lw': 2})
ax.set_xticks([np.log(i + 1) for i in [0, 1, 2, 3, 5, 10, 20, 40]])
ax.set_xticklabels(['0', '1', '2', '3', '5', '10', '20', '40'], fontsize=12)
ax.set_yticks(np.arange(1, 1.7, .1))
ax.set_yticklabels([str(round(i, 1)) for i in np.arange(1, 1.7, .1)])
ax.set_xlabel('TMB', fontsize=16)
ax.set_ylabel('Augmented fold change', fontsize=16)
ax.tick_params(length=3, width=1)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_position(['outward', -5])
ax.spines['left'].set_position(['outward', -5])
ax.spines['left'].set_bounds(1, 1.6)
ax.spines['bottom'].set_bounds(np.log(1 + 0), np.log(1 + 40))

plt.savefig(cwd / 'figures' / 'figure_2' / 'regplot.png', dpi=600)