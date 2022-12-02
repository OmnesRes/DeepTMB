import matplotlib
matplotlib.use('TKAgg')
import pathlib
import sys
from matplotlib import pyplot as plt
path = pathlib.Path.cwd()
if path.stem == 'DeepTMB':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('DeepTMB')]
    sys.path.append(str(cwd))

##if running from command line won't import
try:
    from model import utils
except:
    path_root = pathlib.Path(__file__).parents[1]
    sys.path.append(str(path_root))
    from model import utils

import numpy as np
import tensorflow as tf
from model.model import Encoders, NN
import pandas as pd

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[-2], True)
tf.config.experimental.set_visible_devices(physical_devices[-2], 'GPU')

##read data
data = pd.read_csv(cwd / 'example' / 'example_training_data.txt', sep=',')

##create arrays
X = data['panel_values'].values
Y = data['exome_values'].values

##no weighting in this example, if you care about a certain TMB region can add weights
y_weights = np.ones_like(Y)

##log transform and make loaders
t = utils.LogTransform(bias=4, min_x=0)
X = t.trf(X[:, np.newaxis])
Y = t.trf(Y)
X_loader = utils.Map.PassThrough(X)
Y_loader = utils.Map.PassThrough(Y)
W_loader = utils.Map.PassThrough(y_weights)

##build graph
count_encoder = Encoders.Encoder(shape=(1,), layers=(128,))
net = NN(encoders=[count_encoder.model], layers=(64, 32), mode='mixture')
net.model.compile(loss=utils.log_prob_loss,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.001, patience=30, mode='min', restore_best_weights=True)]

##train on entire dataset
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

##fit model
net.model.fit(ds_train,
              steps_per_epoch=10,
              epochs=10000,
              callbacks=callbacks
              )

##define where to calculate probabilities (should cover range of possible Y distribution)
y_pred = np.linspace(0, np.max(Y + .5), 1000)

##check the benefit of the fit

##get predictions for a range of values(median)
x_pred = np.linspace(np.min(X), np.max(X), 200)
pred_quantiles = [y_pred[np.argmin(np.diff((np.exp(net.model(x_pred).log_cdf(y_pred[:, np.newaxis]).numpy()) < 0.05).astype(int), axis=0), axis=0)],
             y_pred[np.argmin(np.diff((np.exp(net.model(x_pred).log_cdf(y_pred[:, np.newaxis]).numpy()) < 0.25).astype(int), axis=0), axis=0)],
             y_pred[np.argmin(np.diff((np.exp(net.model(x_pred).log_cdf(y_pred[:, np.newaxis]).numpy()) < 0.5).astype(int), axis=0), axis=0)],
             y_pred[np.argmin(np.diff((np.exp(net.model(x_pred).log_cdf(y_pred[:, np.newaxis]).numpy()) < 0.75).astype(int), axis=0), axis=0)],
            y_pred[np.argmin(np.diff((np.exp(net.model(x_pred).log_cdf(y_pred[:, np.newaxis]).numpy()) < 0.95).astype(int), axis=0), axis=0)]]

##make a plot to check fit
fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(top=.98,
bottom=0.108,
left=0.1,
right=.987,
hspace=0.2,
wspace=0.2)
ax.scatter(X, Y, s=5, edgecolor='none', alpha=.3)
for i in range(len(pred_quantiles)):
    if i == 0:
        ax.plot(x_pred, pred_quantiles[i], color='k', linestyle='dashed', alpha=.5, label='Percentiles (5, 25, ' + r'$\bf{50}$' + ', 75, 95)')
    elif i == 2:
        ax.plot(x_pred, pred_quantiles[i], color='k', linewidth=2, linestyle='dashed', alpha=1)
    else:
        ax.plot(x_pred, pred_quantiles[i], color='k', linestyle='dashed', alpha=.5)
ax.set_xticks([t.trf(i) for i in [0, 2, 5, 10, 25, 50]])
ax.set_xticklabels([0, 2, 5, 10, 25, 50])
ax.set_yticks([t.trf(i) for i in [0, 2, 5, 10, 20, 35, 60]])
ax.set_yticklabels([0, 2, 5, 10, 20, 35, 60], fontsize=12)
ax.set_xlabel('Panel-Derived TMB', fontsize=16)
ax.set_ylabel('Exomic TMB', fontsize=16)
ax.tick_params(length=3, width=1)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_position(['outward', 5])
ax.spines['left'].set_bounds(t.trf(0), t.trf(60))
ax.spines['bottom'].set_bounds(t.trf(0), t.trf(50))
plt.legend(frameon=False, loc=(.01, .95), fontsize=10)
ax.set_ylim(t.trf(0), t.trf(60))
ax.set_xlim(-.01, t.trf(50))


##get predictions for all the data (median)
medians = y_pred[np.argmin(np.diff((np.exp(net.model(X).log_cdf(y_pred[:, np.newaxis]).numpy()) < 0.5).astype(int), axis=0), axis=0)]

##mean absolute error for panel TMB (the input to the model)
print(np.mean(np.abs(t.inv(Y) - t.inv(X[:, 0]))))

##mean absolute error for predictions
print(np.mean(np.abs(t.inv(Y) - t.inv(medians))))

##for only panel TMB >= 5
mask = t.inv(X[:, 0]) >= 5
##mean absolute error for panel TMB (the input to the model)
print(np.mean(np.abs(t.inv(Y[mask]) - t.inv(X[:, 0][mask]))))

##mean absolute error for predictions
print(np.mean(np.abs(t.inv(Y[mask]) - t.inv(medians[mask]))))

##load data to predict on
x_pred = t.trf(pd.read_csv(cwd / 'example' / 'example_prediction_data.txt')['panel_data'].values)[:, np.newaxis]

##make predictions
quantiles = [y_pred[np.argmin(np.diff((np.exp(net.model(x_pred).log_cdf(y_pred[:, np.newaxis]).numpy()) < 0.05).astype(int), axis=0), axis=0)],
             y_pred[np.argmin(np.diff((np.exp(net.model(x_pred).log_cdf(y_pred[:, np.newaxis]).numpy()) < 0.25).astype(int), axis=0), axis=0)],
             y_pred[np.argmin(np.diff((np.exp(net.model(x_pred).log_cdf(y_pred[:, np.newaxis]).numpy()) < 0.5).astype(int), axis=0), axis=0)],
             y_pred[np.argmin(np.diff((np.exp(net.model(x_pred).log_cdf(y_pred[:, np.newaxis]).numpy()) < 0.75).astype(int), axis=0), axis=0)],
            y_pred[np.argmin(np.diff((np.exp(net.model(x_pred).log_cdf(y_pred[:, np.newaxis]).numpy()) < 0.95).astype(int), axis=0), axis=0)]]

##write predictions
pd.DataFrame(data={'5th percentile': t.inv(quantiles[0]),
                   '25th percentile': t.inv(quantiles[1]),
                   '50th percentile': t.inv(quantiles[2]),
                   '75th percentile': t.inv(quantiles[3]),
                   '95th percentile': t.inv(quantiles[4])}).to_csv(cwd / 'example' / 'example_predictions.txt', index=False, sep='\t')
