import numpy as np
import tensorflow as tf
from model.model import Encoders, NN
from model import utils
import pickle
from sklearn.model_selection import StratifiedKFold

import pathlib
path = pathlib.Path.cwd()
if path.stem == 'DeepTMB':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('DeepTMB')]
    import sys
    sys.path.append(str(cwd))

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[3], True)
tf.config.experimental.set_visible_devices(physical_devices[3], 'GPU')

data = pickle.load(open(cwd / 'tables' / 'table_1' / 'DUKE-F1-DX1' / 'tumor_normal' / 'data' / 'data.pkl', 'rb'))
ancestry = pickle.load(open(cwd / 'files' / 'ethnicity.pkl', 'rb'))

[data.pop(i) for i in list(data.keys()) if not data[i]]

non_syn = ['Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins', 'In_Frame_Del', 'In_Frame_Ins', 'Nonstop_Mutation']
non_syn_data = {i: sum([data[i][5].to_dict()[j] for j in data[i][5].index if j in non_syn]) for i in data}
cutoff = np.percentile(list(non_syn_data.values()), 98)
values = [data[i] for i in data if non_syn_data[i] < cutoff]
non_syn_counts = [i for i in non_syn_data.values() if i < cutoff]
anc = np.array([ancestry.get(i[:12], 'OA') for i in non_syn_data if non_syn_data[i] < cutoff])
anc_encoding = {'AA': 1, 'EA': 2, 'EAA': 3, 'NA': 4, 'OA': 0}
anc = np.array([anc_encoding[i] for i in anc])

X = np.array([i / (j[1] / 1e6) for i, j in zip(non_syn_counts, values)])
Y = np.array([i[2] / (i[3] / 1e6) for i in values])

t = utils.LogTransform(bias=4, min_x=0)
X = t.trf(X[:, np.newaxis])
Y = t.trf(Y)
X_loader = utils.Map.PassThrough(X)
Y_loader = utils.Map.PassThrough(Y)

count_encoder = Encoders.Encoder(shape=(1,), layers=())
net = NN(encoders=[count_encoder.model], layers=(), mode='tfp_linear_regresion')
net.model.compile(loss=utils.log_prob_loss,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

weights = net.model.get_weights()
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=30, mode='min', restore_best_weights=True)]

predictions = []
losses = []
test_idx = []
y_pred = np.linspace(np.min(Y), np.max(Y + .5), 1000)
for idx_train, idx_test in StratifiedKFold(n_splits=5, random_state=0, shuffle=True).split(anc, anc):
    net.model.set_weights(weights)
    test_idx.append(idx_test)
    ds_train = tf.data.Dataset.from_tensor_slices((idx_train,))
    ds_train = ds_train.shuffle(buffer_size=len(anc), reshuffle_each_iteration=True).repeat().batch(batch_size=int(len(idx_train) * .75), drop_remainder=True)
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
    predictions.append(y_pred[np.argmin(np.diff((np.exp(net.model(X[idx_test]).log_cdf(y_pred[:, np.newaxis]).numpy()) < 0.5).astype(int), axis=0), axis=0)])
    losses.append(net.model.evaluate(X[idx_test], Y[idx_test]))


with open(cwd / 'tables' / 'table_1' / 'DUKE-F1-DX1' / 'tumor_normal' / 'results' / 'linear_prob_nonsyn_predictions.pkl', 'wb') as f:
    pickle.dump([predictions, test_idx, values, losses], f)

##check each fold trained
for fold, preds in zip(test_idx, predictions):
    print(np.mean((t.inv(preds) - t.inv(Y[fold])) ** 2)**.5)

