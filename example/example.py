import numpy as np
import tensorflow as tf
from model.model import Encoders, NN
from model import utils
import pandas as pd

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

##read data
data = pd.read_csv(cwd / 'example' / 'example_training_data.txt', sep='\t')

##create arrays
X = data['panel_values'].values
Y = data['exome_values'].values

##no weighting in this example
y_weights = np.ones_like(Y)

##log transform and make loaders
t = utils.LogTransform(bias=4, min_x=0)
X = t.trf(X[:, np.newaxis])
Y = t.trf(Y)
X_loader = utils.Map.PassThrough(X)
Y_loader = utils.Map.PassThrough(Y)
Y_loader_fcn = utils.Map.PassThrough(Y[:, np.newaxis])
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

##check the benefit of the fit

##define where to calculate probabilities
y_pred = np.linspace(0, np.max(Y + .5), 1000)

##get median predictions
medians = y_pred[np.argmin(np.diff((np.exp(net.model(X).log_cdf(y_pred[:, np.newaxis]).numpy()) < 0.5).astype(int), axis=0), axis=0)]

##mean absolute error in original data space prior to fit
print(np.mean(np.abs(t.inv(Y) - t.inv(X[:, 0]))))

##mean absolute error in original data space
print(np.mean(np.abs(t.inv(Y) - t.inv(medians))))

##for only panel TMB >= 5
mask = t.inv(X[:, 0]) >= 5
##mean absolute error in original data space prior to fit
print(np.mean(np.abs(t.inv(Y[mask]) - t.inv(X[:, 0][mask]))))

##mean absolute error in original data space
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





