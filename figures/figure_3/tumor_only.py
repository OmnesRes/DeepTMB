import numpy as np
from model.model import Encoders, NN, Losses
from model import utils
import pickle
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.preprocessing import OneHotEncoder
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
tf.config.experimental.set_memory_growth(physical_devices[-1], True)
tf.config.experimental.set_visible_devices(physical_devices[-1], 'GPU')

##use MSK 468 data
sample_table = pickle.load(open(cwd / 'files' / 'tcga_public_sample_table.pkl', 'rb'))
data = pickle.load(open(cwd / 'figures' / 'figure_3' / 'data' / 'data.pkl', 'rb'))
germline_samples = pickle.load(open(cwd / 'files' / 'germline' / 'data' / 'germline_samples.pkl', 'rb'))

[germline_samples.pop(i) for i in list(germline_samples.keys()) if germline_samples[i] < 400]
[data.pop(i) for i in list(data.keys()) if not data[i]]
[data.pop(i) for i in list(data.keys()) if i[:12] not in germline_samples]
nci_dict = {i: j for i, j in zip(sample_table['Tumor_Sample_Barcode'].values, sample_table['NCIt_tmb_label'].values) if j}
[data.pop(i) for i in list(data.keys()) if i not in nci_dict]

values = [i for i in data.values() if (i[2] / (i[3] / 1e6)) <= 40]
nci = np.array([nci_dict[i] for i in data if (data[i][2] / (data[i][3] / 1e6)) <= 40])
class_counts = dict(zip(*np.unique(nci, return_counts=True)))
mask = [class_counts[i] >= 50 for i in nci]
nci = nci[mask]

ppvs = [[], [], []]
rmses = [[], [], []]
for i in range(5):
    count_encoder = Encoders.Encoder(shape=(1,), layers=(64,))
    net = NN(encoders=[count_encoder.model], layers=(32, 16))

    net.model.compile(loss=Losses.Normal(name='normal'),
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

    weights = net.model.get_weights()
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=30, mode='min', restore_best_weights=True)]

    for index, filters in enumerate(['loose', 'moderate', 'strict']):
        tumor_only_values = [data[i][0][index] for i in data if (data[i][2] / (data[i][3] / 1e6)) <= 40]
        X = np.array([j / (i[1] / 1e6) for i, j in zip(values, tumor_only_values)])
        Y = np.array([i[2] / (i[3] / 1e6) for i in values])

        X = X[mask, np.newaxis]
        Y = Y[mask, np.newaxis]
        X_loader = utils.Map.PassThrough(X)
        Y_loader = utils.Map.PassThrough(Y)

        cancers_onehot = OneHotEncoder().fit(nci[:, np.newaxis]).transform(nci[:, np.newaxis]).toarray()
        y_strat = np.argmax(cancers_onehot, axis=-1)

        net.model.set_weights(weights)
        idx_train = np.arange(len(X))
        ds_train = tf.data.Dataset.from_tensor_slices((idx_train, ))
        ds_train = ds_train.shuffle(buffer_size=len(y_strat), reshuffle_each_iteration=True).repeat().batch(batch_size=int(len(idx_train) * .75), drop_remainder=True)
        ds_train = ds_train.map(lambda x: ((
                                            X_loader(x),
                                            ),
                                           (Y_loader(x),
                                            )
                                           )
                                )

        net.model.fit(ds_train,
                      steps_per_epoch=10,
                      epochs=10000,
                      callbacks=callbacks
                      )

        x_pred = np.linspace(np.min(X), np.max(X), 200)
        y_pred = net.model.predict(x_pred)
        Y_pred = net.model.predict(X)

        y_preds = tfp.distributions.Normal(loc=y_pred[:, 0], scale=np.exp(y_pred[:, 1]))
        Y_preds = tfp.distributions.Normal(loc=Y_pred[:, 0], scale=np.exp(Y_pred[:, 1]))
        lower_estimates = Y_preds.quantile(.16).numpy()
        upper_estimates = Y_preds.quantile(.84).numpy()

        ##check PPV
        ppvs[index].append(sum((Y_preds.mean().numpy() >= 10) & (Y[:, 0] >= 10)) / sum(Y_preds.mean().numpy() >= 10))
        rmses[index].append(np.sqrt(np.mean((Y_preds.mean() - Y[:, 0])**2)))
        if i == 0:
            ##input output plot
            fig = plt.figure()
            ax = fig.add_subplot(111)
            fig.subplots_adjust(top=1,
            bottom=0.084,
            left=0.093,
            right=1.0,
            hspace=0.2,
            wspace=0.2)
            ax.scatter(np.log(X + 1), np.log(Y + 1), s=5, edgecolor='none', alpha=.15)
            ax.plot(np.log(x_pred + 1), np.log(y_preds.mean() + 1), color='k', linestyle='-', alpha=.5)
            ax.plot(np.log(x_pred + 1), np.log(y_preds.quantile(.84) + 1), color='k', linestyle='dashed', alpha=.3)
            ax.plot(np.log(x_pred + 1), np.log(y_preds.quantile(.16) + 1), color='k', linestyle='dashed', alpha=.3)
            ax.set_xticks([np.log(i+1) for i in range(101)])
            ax.set_xticklabels([])
            ax.set_yticks([np.log(i + 1) for i in [0, 1, 2, 3, 5, 10, 20, 40]])
            ax.set_yticklabels(['0', '1', '2', '3', '5', '10', '20', '40'], fontsize=12)
            ax.set_xlabel('Input Values', fontsize=16)
            ax.set_ylabel('TMB', fontsize=16)
            ax.tick_params(length=3, width=1)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_position(['outward', 2])
            ax.spines['left'].set_position(['outward', 2])
            ax.spines['left'].set_bounds(np.log(1 + 0), np.log(1 + 40))
            ax.spines['bottom'].set_bounds(np.log(1 + 0), np.log(1 + 100))
            ax.set_xlim(-.1, np.log(101))
            ax.set_ylim(0, np.log(44))
            plt.savefig(cwd / 'figures' / 'figure_3' / 'results' / ('germline_' + filters + '_input_output.png'), dpi=600)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            fig.subplots_adjust(top=1,
            bottom=0.105,
            left=0.093,
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
            ax.spines['bottom'].set_position(['outward', 2])
            ax.spines['left'].set_position(['outward', 2])
            ax.set_xlim(0, np.log(44))
            ax.set_ylim(0, np.log(44))
            plt.savefig(cwd / 'figures' / 'figure_3' / 'results' / ('germline_' + filters + '_pred_true.png'), dpi=600)

##mean ppvs: [0.75348412, 0.78400372, 0.82243839]
##mean rmses: [3.469481 , 3.247291 , 2.9772243]