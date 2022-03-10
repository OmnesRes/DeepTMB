import numpy as np
from model.model import Encoders, NN, Losses
from model import utils
import pickle
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

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
data = pickle.load(open(cwd / 'tables' / 'table_1' / 'MSK_468' / 'data' / 'data.pkl', 'rb'))
distributions = pickle.load(open(cwd / 'figures' / 'figure_2' / 'distributions.pkl', 'rb'))

for i in list(data.keys()):
    if i not in distributions:
        data.pop(i)
    else:
        if not distributions[i]:
            data.pop(i)


##this table was limited to samples that had TMB less than 40
nci_table = pd.read_csv(open(cwd / 'files' / 'NCI-T.tsv'), sep='\t').dropna()
nci_dict = {i: j for i, j in zip(nci_table['Tumor_Sample_Barcode'].values, nci_table['NCI-T Label TMB'].values)}

result = data.copy()
[result.pop(i) for i in data if i not in nci_dict]
values = [i for i in result.values() if i]

X = np.array([i[0] / (i[1] / 1e6) for i in values])
Y = np.array([i[2] / (i[3] / 1e6) for i in values])

X_distributions = np.array([np.array(distributions[i][0]) / (result[i][1] / 1e6) for i in result if result[i]])
Y_distributions = np.array([np.array(distributions[i][1]) / (result[i][3] / 1e6) for i in result if result[i]])

nci = np.array([nci_dict[i] for i in result if result[i]])

class_counts = dict(zip(*np.unique(nci, return_counts=True)))
mask = [class_counts[i] >= 50 for i in nci]

nci = nci[mask]
X = X[mask, np.newaxis]
Y = Y[mask, np.newaxis]
X_distributions = X_distributions[mask]
Y_distributions = Y_distributions[mask]

cancers_onehot = OneHotEncoder().fit(nci[:, np.newaxis]).transform(nci[:, np.newaxis]).toarray()
y_strat = np.argmax(cancers_onehot, axis=-1)

count_encoder = Encoders.Encoder(shape=(1,), layers=(64,))
net = NN(encoders=[count_encoder.model], layers=(32, 16))

net.model.compile(loss=Losses.LogNormal(name='lognormal'),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

weights = net.model.get_weights()
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=30, mode='min', restore_best_weights=True)]



##unaugmented

X_loader = utils.Map.PassThrough(X)
Y_loader = utils.Map.PassThrough(Y)

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


with open(cwd / 'figures' / 'figure_2' / 'unaugmented_fit.pkl', 'wb') as f:
    pickle.dump([x_pred, y_pred, X, Y, Y_pred], f)


##average augmented

X_loader = utils.Map.PassThrough(np.mean(X_distributions, axis=-1, keepdims=True))
Y_loader = utils.Map.PassThrough(np.mean(Y_distributions, axis=-1, keepdims=True))

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

x_pred = np.linspace(np.min(np.mean(X_distributions, axis=-1, keepdims=True)), np.max(np.mean(X_distributions, axis=-1, keepdims=True)), 200)
y_pred = net.model.predict(x_pred)
Y_pred = net.model.predict(np.mean(X_distributions, axis=-1, keepdims=True))


with open(cwd / 'figures' / 'figure_2' / 'augmented_fit_once.pkl', 'wb') as f:
    pickle.dump([x_pred, y_pred, X_distributions, Y_distributions, Y_pred], f)


##augmented per batch

X_loader = utils.Map.Augment(X_distributions)
Y_loader = utils.Map.Augment(Y_distributions)


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

x_pred = np.linspace(np.min(X_distributions), np.max(X_distributions), 200)
y_pred = net.model.predict(x_pred)
Y_pred = net.model.predict(np.mean(X_distributions, axis=-1))


with open(cwd / 'figures' / 'figure_2' / 'augmented_fit.pkl', 'wb') as f:
    pickle.dump([x_pred, y_pred, X_distributions, Y_distributions, Y_pred], f)
