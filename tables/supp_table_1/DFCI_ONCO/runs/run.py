import numpy as np
from model.model import Encoders, NN, Losses
from model import utils
import pickle
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
import tensorflow_probability as tfp

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

pcawg_maf = pickle.load(open(path / 'files' / 'pcawg_maf_table.pkl', 'rb'))
samples = pickle.load(open(path / 'files' / 'pcawg_sample_table.pkl', 'rb'))
panels = pickle.load(open(path / 'files' / 'pcawg_panel_table.pkl', 'rb'))

samples = samples.loc[samples['non_syn_counts'] > 0]

pcawg_maf = pcawg_maf.loc[pcawg_maf['DFCI-ONCOPANEL-3'] > 0]
panel_counts = pcawg_maf[['Tumor_Sample_Barcode']].groupby('Tumor_Sample_Barcode').apply(lambda x: pd.Series([len(x)], index=['panel_counts']))
panel_counts.reset_index(inplace=True)
samples = pd.merge(samples, panel_counts, how='inner', left_on='aliquot_id', right_on='Tumor_Sample_Barcode')

##histology embedding
samples['cancer_code'] = samples['project_code'].apply(lambda x: x.split('-')[0])
cancer_dict = {'LICA': 'liver', 'LINC': 'liver', 'LIRI': 'liver', 'BTCA': 'liver', 'BOCA': 'bone', 'BRCA': 'breast',
               'CLLE': 'blood', 'CMDI': 'blood', 'LAML': 'blood', 'MALY': 'blood', 'EOPC': 'prostate', 'PRAD': 'prostate',
               'OV': 'ovarian', 'MELA': 'skin', 'ESAD':'orogastric', 'ORCA': 'orogastric', 'GACA': 'orogastric',
               'PBCA': 'brain', 'RECA': 'renal', 'PAEN': 'pancreas', 'PACA': 'pancreas'}

samples['cancer'] = samples['cancer_code'].apply(lambda x: cancer_dict[x])
A = samples['cancer'].astype('category')
classes = A.cat.categories.values
classes_onehot = np.eye(len(classes))[A.cat.codes]
y_strat = np.argmax(classes_onehot, axis=-1)

X = np.array([i / (panels.loc[panels['Panel'] == 'DFCI-ONCOPANEL-3']['total'].values[0] / 1e6) for i in samples['panel_counts'].values])
Y = np.array([i / (panels.loc[panels['Panel'] == 'CDS']['cds'].values[0] / 1e6) for i in samples['non_syn_counts'].values])

mask = Y < 40
y_strat = y_strat[mask]
X = np.log(X[mask, np.newaxis] + 1)
Y = np.log(Y[mask, np.newaxis] + 1)
X_loader = utils.Map.PassThrough(X)
Y_loader = utils.Map.PassThrough(Y)

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

with open(cwd / 'tables' / 'supp_table_1' / 'DFCI_ONCO' / 'results' / 'run_predictions.pkl', 'wb') as f:
    pickle.dump([predictions, test_idx, [X, Y]], f)

##check each fold trained
for fold, preds in zip(test_idx, predictions):
    print(np.mean((preds - Y[fold][:, 0]) ** 2))

np.mean((np.log(np.concatenate(predictions) + 1) - np.log(Y[:, 0][np.concatenate(test_idx)] + 1))**2)
np.mean((np.concatenate(predictions) - Y[:, 0][np.concatenate(test_idx)])**2)
