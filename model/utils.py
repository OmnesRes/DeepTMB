import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

class Apply:

    class StratifiedMinibatch:
        def __init__(self, batch_size, ds_size, reshuffle_each_iteration=True):
            self.batch_size, self.ds_size, self.reshuffle_each_iteration = batch_size, ds_size, reshuffle_each_iteration
            # max number of splits
            self.n_splits = (self.ds_size // self.batch_size) + 1
            # stratified "mini-batch" via k-fold
            self.batcher = StratifiedKFold(n_splits=self.n_splits, shuffle=self.reshuffle_each_iteration)

        def __call__(self, ds_input: tf.data.Dataset):
            def generator():
                # expecting ds of (idx, y_strat)
                idx, y_strat = list(map(tf.stack, list(map(list, zip(*list(ds_input))))))
                while True:
                    for _, batch_idx in self.batcher.split(y_strat, y_strat):
                        yield tf.gather(idx, batch_idx, axis=0)

            return tf.data.Dataset.from_generator(generator,
                                                  output_types=(ds_input.element_spec[0].dtype),
                                                  output_shapes=((None, )))

    class StratifiedBootstrap:
        def __init__(self, batch_class_sizes=[]):
            self.batch_class_sizes = batch_class_sizes
            self.batch_size = sum(self.batch_class_sizes)
            self.rnd = tf.random.Generator.from_non_deterministic_state()

        def __call__(self, ds_input: tf.data.Dataset):
            def generator():
                # expecting ds of (idx, y_strat)
                idx, y_strat = list(map(tf.stack, list(map(list, zip(*list(ds_input))))))
                assert (tf.reduce_max(y_strat).numpy() + 1) == len(self.batch_class_sizes)
                class_idx = [tf.where(y_strat == i)[:, 0] for i in range(len(self.batch_class_sizes))]
                while True:
                    batch_idx = list()
                    for j in range(len(self.batch_class_sizes)):
                        batch_idx.append(tf.gather(class_idx[j], self.rnd.uniform(shape=(self.batch_class_sizes[j], ),
                                                                                  maxval=tf.cast(class_idx[j].shape[0] - 1, tf.int64),
                                                                                  dtype=tf.int64)))
                    batch_idx = tf.concat(batch_idx, axis=0)

                    yield tf.gather(idx, batch_idx, axis=0)

            return tf.data.Dataset.from_generator(generator,
                                                  output_types=(ds_input.element_spec[0].dtype),
                                                  output_shapes=((self.batch_size)))

    class SubSample:
        def __init__(self, batch_size, ds_size):
            self.batch_size = batch_size
            self.ds_size = ds_size
        def __call__(self, ds_input: tf.data.Dataset):
            def generator():
                # expecting ds of idx
                idx = tf.stack([element for element in ds_input])
                while True:
                    batch_idx = np.random.choice(np.arange(self.ds_size), self.batch_size, replace=False)
                    yield tf.gather(idx, batch_idx, axis=0)

            return tf.data.Dataset.from_generator(generator,
                                                  output_types=(ds_input.element_spec.dtype),
                                                  output_shapes=((None, )))

class Map:
    class LoadBatch:
        def loader(self):
            raise NotImplementedError

        def __call__(self, sample_idx):
            flat_values, *additional_args = tf.py_function(self.loader, [sample_idx], [tf.float32])
            return flat_values

    class Augment(LoadBatch):
        def __init__(self, data):
            self.data = data

        def loader(self, idx):
            choice = np.random.choice(np.arange(self.data.shape[-1]))
            return self.data[idx, choice][:, np.newaxis]
            # batch.append(np.random.choice(self.data[i]))
            # return np.array(batch)[:, np.newaxis]

    class PassThrough(LoadBatch):
        def __init__(self, data):
            self.data = data

        def loader(self, idx):
            return self.data[idx]
