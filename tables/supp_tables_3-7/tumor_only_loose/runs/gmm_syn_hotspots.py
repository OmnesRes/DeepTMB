import numpy as np
import tensorflow as tf
from model.model import Encoders, NN
from model import utils
import pickle

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

data = pickle.load(open(cwd / 'tables' / 'table_1' / 'DUKE-F1-DX1' / 'tumor_only_loose' / 'data' / 'data.pkl', 'rb'))
ancestry = pickle.load(open(cwd / 'files' / 'ethnicity.pkl', 'rb'))
germline_samples = pickle.load(open(cwd / 'files' / 'germline' / 'data' / 'germline_samples.pkl', 'rb'))

[data.pop(i) for i in list(data.keys()) if not data[i]]
[germline_samples.pop(i) for i in list(germline_samples.keys()) if germline_samples[i] < 400]
[data.pop(i) for i in list(data.keys()) if i[:12] not in germline_samples]

cutoff = np.percentile([i[-1] / (i[1] / 1e6) for i in data.values()], 98)
mask = [(i[-1] / (i[1] / 1e6)) < cutoff for i in data.values()]
anc = np.array([ancestry.get(i[:12], 'OA') for i in data])
anc_encoding = {'AA': 1, 'EA': 2, 'EAA': 3, 'NA': 4, 'OA': 0}
anc = np.array([anc_encoding[i] for i in anc])
anc = anc[mask]
X = np.array([(i[0] - i[4]) / (i[1] / 1e6) for i in data.values()])
Y = np.array([i[2] / (i[3] / 1e6) for i in data.values()])

t = utils.LogTransform(bias=4, min_x=0)
X = t.trf(X[mask, np.newaxis])
Y = t.trf(Y[mask])

for ancestry in range(1, 6):
    if ancestry == 5:
        anc_mask = anc < 100
    else:
        anc_mask = anc == ancestry
    X_loader = utils.Map.PassThrough(X[anc_mask])
    Y_loader = utils.Map.PassThrough(Y[anc_mask])

    count_encoder = Encoders.Encoder(shape=(1,), layers=(128,))
    net = NN(encoders=[count_encoder.model], layers=(64, 32), mode='mixture')
    net.model.compile(loss=utils.log_prob_loss,
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=30, mode='min', restore_best_weights=True)]
    idx_train = np.arange(len(X[anc_mask]))
    ds_train = tf.data.Dataset.from_tensor_slices((idx_train,))
    ds_train = ds_train.shuffle(buffer_size=len(idx_train), reshuffle_each_iteration=True).repeat().batch(batch_size=int(len(idx_train) * .75), drop_remainder=True)
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
    y_pred = np.linspace(0, np.max(Y + 2), 1000)
    predictions = y_pred[np.argmin(np.diff((np.exp(net.model(X[anc_mask]).log_cdf(y_pred[:, np.newaxis]).numpy()) < 0.5).astype(int), axis=0), axis=0)]
    losses = net.model.evaluate(X[anc_mask], Y[anc_mask])

    with open(cwd / 'tables' / 'supp_tables_3-7' / 'tumor_only_loose' / 'results' / ('gmm_syn_hotspots_predictions_' + str(ancestry) + '.pkl'), 'wb') as f:
        pickle.dump([predictions, [X[anc_mask], Y[anc_mask]], losses], f)
