import numpy as np
from model.model import Encoders, NN, Losses
from model import utils
import pickle
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
import tensorflow_probability as tfp
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

data = pickle.load(open(cwd / 'tables' / 'table_1' / 'MSK_341' / 'data' / 'data.pkl', 'rb'))
##this table was limited to samples that had TMB less than 40
nci_table = pd.read_csv(open(cwd / 'files' / 'NCI-T.tsv'), sep='\t').dropna()
nci_dict = {i: j for i, j in zip(nci_table['Tumor_Sample_Barcode'].values, nci_table['NCI-T Label TMB'].values)}

result = data.copy()
[result.pop(i) for i in data if i not in nci_dict]

values = [i for i in result.values() if i]
non_syn = ['Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins', 'In_Frame_Del', 'In_Frame_Ins', 'Nonstop_Mutation']
non_syn_counts = [sum([i[5].to_dict()[j] for j in i[5].index if j in non_syn]) for i in values]
X = np.array([i / (j[1] / 1e6) for i, j in zip(non_syn_counts, values)])
Y = np.array([i[2] / (i[3] / 1e6) for i in values])

nci = np.array([nci_dict[i] for i in result if result[i]])

class_counts = dict(zip(*np.unique(nci, return_counts=True)))

mask = [class_counts[i] >= 50 for i in nci]
nci = nci[mask]
X = X[mask, np.newaxis]
Y = Y[mask, np.newaxis]
X_loader = utils.Map.PassThrough(X)
Y_loader = utils.Map.PassThrough(Y)

nci_onehot = OneHotEncoder().fit(nci[:, np.newaxis]).transform(nci[:, np.newaxis]).toarray()
y_strat = np.argmax(nci_onehot, axis=-1)

count_encoder = Encoders.Encoder(shape=(1,), layers=(64,))
net = NN(encoders=[count_encoder.model], layers=(32, 16))

net.model.compile(loss=Losses.LogNormal(name='lognormal'),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

weights = net.model.get_weights()
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=30, mode='min', restore_best_weights=True)]

predictions = []
test_idx = []
for idx_train, idx_test in StratifiedKFold(n_splits=5, random_state=0, shuffle=True).split(y_strat, y_strat):
    net.model.set_weights(weights)
    test_idx.append(idx_test)
    ds_train = tf.data.Dataset.from_tensor_slices((idx_train,))
    ds_train = ds_train.shuffle(buffer_size=len(y_strat), reshuffle_each_iteration=True).repeat().batch(batch_size=int(len(idx_train) * .75), drop_remainder=True)
    ds_train = ds_train.map(lambda x: ((
                                        X_loader(x),

                                        ),
                                       (Y_loader(x),
                                        )
                                       )
                            )

    ds_test = tf.data.Dataset.from_tensor_slices((
                                                 (X[idx_test],),
                                                 ))
    ds_test = ds_test.batch(len(idx_test), drop_remainder=False)
    net.model.fit(ds_train,
                  steps_per_epoch=10,
                  epochs=10000,
                  callbacks=callbacks
                  )
    pred = net.model.predict(ds_test)
    predictions.append(tfp.distributions.LogNormal(loc=pred[:, 0], scale=np.exp(pred[:, 1])).mean().numpy())

with open(cwd / 'tables' / 'table_1' / 'MSK_341' / 'results' / 'run_nonsyn_predictions.pkl', 'wb') as f:
    pickle.dump([predictions, test_idx, values], f)

##check each fold trained
for fold, preds in zip(test_idx, predictions):
    print(np.mean((preds - Y[fold][:, 0]) ** 2))


