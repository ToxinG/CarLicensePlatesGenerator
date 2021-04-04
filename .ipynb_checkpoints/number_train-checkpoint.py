import tensorflow as tf
from tensorflow.keras import layers

import numpy as np
import pandas as pd
from skimage import io, data, filters
from skimage.transform import rescale, resize

import os
from os.path import join
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-g', '--gpu', required=True, type=str, help='gpu to use')
ap.add_argument('-v', '--version', required=True, type=str, help='version name')
ap.add_argument('-e', '--epochs', default=20, type=int, help='number of epochs')
ap.add_argument('-b', '--batch', default=16, type=int, help='size of batch')

args = vars(ap.parse_args())
gpu = args['gpu']
model_name = args['version']
epochs = args['epochs']
batch_size = args['batch']

# **Select gpu device**

os.environ["CUDA_VISIBLE_DEVICES"] = gpu

print('tf version: ', tf.__version__)
if tf.__version__ == '1.13.1':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.keras.backend.set_session(tf.Session(config=config))
elif tf.__version__ == '2.0.0':
    tf.config.gpu.set_per_process_memory_growth(True)

# # Data
# ## Load meta
meta_folder = '/media/disk1/markup'
data_folder = '/media/disk1/data'

train_df_names = ['plate_train.csv']
test_df_names = ['plate_test.csv']


def load_and_concat_df(df_names):
    dfs = [pd.read_csv(join(meta_folder, name)) for name in df_names]
    return pd.concat(dfs).reset_index(drop=True)


train_df = load_and_concat_df(train_df_names)
test_df = load_and_concat_df(test_df_names)


# ## Load data
def load_from_df(df):
    images = [io.imread(path) for path in df['path']]
    plates = df['plate']
    return images, plates


train_images, train_plates = load_from_df(train_df)
test_images, test_plates = load_from_df(test_df)

input_shape = (40, 160, 3)


def prepocess_images(images, target_shape=input_shape):
    return np.array([resize(image, target_shape) / 255. for image in images])


# ## Encode Y(plates) to one_hot
alphabet = '0123456789ABCEHKMOPTXY_'


def _one_hot(i, n=len(alphabet)):
    res = np.zeros(n)
    res[i] = 1
    return res


def _encode_plate_one_hot(plate: str):
    if len(plate) == 8:
        plate += '_'
    return np.array([_one_hot(alphabet.find(x)) for x in plate])


def encode_plates(plates):
    return np.array([_encode_plate_one_hot(plate) for plate in plates])


# ## Eval X, y
X, y = prepocess_images(train_images), encode_plates(train_plates)
X_test, y_test = prepocess_images(test_images), encode_plates(test_plates)


# ## Define loss
def plate_loss(y_true, y_pred):
    n = 9
    y_pred = tf.transpose(y_pred, perm=[1, 0, 2])
    y_true = tf.transpose(y_true, perm=[1, 0, 2])
    losses = [tf.keras.losses.categorical_crossentropy(y_true[i], y_pred[i]) / n for i in range(n)]
    total_loss = tf.math.reduce_sum(tf.convert_to_tensor(losses))
    return total_loss


# ## Define metrics
def plate_acc(y_true, y_pred):
    n = 9
    y_pred = tf.transpose(y_pred, perm=[1, 0, 2])
    y_true = tf.transpose(y_true, perm=[1, 0, 2])
    acc_fn = tf.keras.metrics.CategoricalAccuracy()
    acc = [acc_fn(y_true[i], y_pred[i]) / n for i in range(n)]
    total_acc = tf.math.reduce_sum(tf.convert_to_tensor(acc))
    return total_acc


# ## Define model
def m1_conv_32():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(filters=32, kernel_size=(2, 2), input_shape=input_shape, padding='same'))
    model.add(layers.MaxPool2D())
    model.add(layers.Conv2D(filters=32, kernel_size=(2, 2), padding='same'))
    model.add(layers.MaxPool2D())
    model.add(layers.Conv2D(filters=9 * 10, kernel_size=(2, 2), padding='same'))
    units = np.prod(model.layers[-1].output_shape[1:])
    assert units % 9 == 0, 'inconsistent shape'
    model.add(layers.Reshape((9, units // 9)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    return model


def m1_conv_64():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(filters=64, kernel_size=(2, 2), input_shape=input_shape, padding='same'))
    model.add(layers.MaxPool2D())
    model.add(layers.Conv2D(filters=64, kernel_size=(2, 2), padding='same'))
    model.add(layers.MaxPool2D())
    model.add(layers.Conv2D(filters=9 * 10, kernel_size=(2, 2), padding='same'))
    units = np.prod(model.layers[-1].output_shape[1:])
    assert units % 9 == 0, 'inconsistent shape'
    model.add(layers.Reshape((9, units // 9)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    return model


def m1_conv_128():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(filters=128, kernel_size=(2, 2), input_shape=input_shape, padding='same'))
    model.add(layers.MaxPool2D())
    model.add(layers.Conv2D(filters=128, kernel_size=(2, 2), padding='same'))
    model.add(layers.MaxPool2D())
    model.add(layers.Conv2D(filters=9 * 10, kernel_size=(2, 2), padding='same'))
    units = np.prod(model.layers[-1].output_shape[1:])
    assert units % 9 == 0, 'inconsistent shape'
    model.add(layers.Reshape((9, units // 9)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    return model


def m1_conv_64_128_256():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(filters=64, kernel_size=(2, 2), input_shape=input_shape, padding='same'))
    model.add(layers.MaxPool2D())
    model.add(layers.Conv2D(filters=128, kernel_size=(2, 2), input_shape=input_shape, padding='same'))
    model.add(layers.MaxPool2D())
    model.add(layers.Conv2D(filters=256, kernel_size=(2, 2), padding='same'))
    model.add(layers.MaxPool2D())
    model.add(layers.Conv2D(filters=9 * 10, kernel_size=(2, 2), padding='same'))
    units = np.prod(model.layers[-1].output_shape[1:])
    assert units % 9 == 0, 'inconsistent shape'
    model.add(layers.Reshape((9, units // 9)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    return model


def m2_dense_32():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(filters=64, kernel_size=(2, 2), input_shape=input_shape, padding='same'))
    model.add(layers.MaxPool2D())
    model.add(layers.Conv2D(filters=128, kernel_size=(2, 2), padding='same'))
    model.add(layers.MaxPool2D())
    model.add(layers.Conv2D(filters=9 * 10, kernel_size=(2, 2), padding='same'))
    units = np.prod(model.layers[-1].output_shape[1:])
    assert units % 9 == 0, 'inconsistent shape'
    model.add(layers.Reshape((9, units // 9)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.3))
    return model


def m2_dense_64():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(filters=64, kernel_size=(2, 2), input_shape=input_shape, padding='same'))
    model.add(layers.MaxPool2D())
    model.add(layers.Conv2D(filters=128, kernel_size=(2, 2), padding='same'))
    model.add(layers.MaxPool2D())
    model.add(layers.Conv2D(filters=9 * 10, kernel_size=(2, 2), padding='same'))
    units = np.prod(model.layers[-1].output_shape[1:])
    assert units % 9 == 0, 'inconsistent shape'
    model.add(layers.Reshape((9, units // 9)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    return model


def m2_dense_128():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(filters=64, kernel_size=(2, 2), input_shape=input_shape, padding='same'))
    model.add(layers.MaxPool2D())
    model.add(layers.Conv2D(filters=128, kernel_size=(2, 2), padding='same'))
    model.add(layers.MaxPool2D())
    model.add(layers.Conv2D(filters=9 * 10, kernel_size=(2, 2), padding='same'))
    units = np.prod(model.layers[-1].output_shape[1:])
    assert units % 9 == 0, 'inconsistent shape'
    model.add(layers.Reshape((9, units // 9)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))
    return model


def m2_dense_256_drop_03():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(filters=64, kernel_size=(2, 2), input_shape=input_shape, padding='same'))
    model.add(layers.MaxPool2D())
    model.add(layers.Conv2D(filters=128, kernel_size=(2, 2), padding='same'))
    model.add(layers.MaxPool2D())
    model.add(layers.Conv2D(filters=9 * 10, kernel_size=(2, 2), padding='same'))
    units = np.prod(model.layers[-1].output_shape[1:])
    assert units % 9 == 0, 'inconsistent shape'
    model.add(layers.Reshape((9, units // 9)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.3))
    return model


def m2_dense_256_drop_04():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(filters=64, kernel_size=(2, 2), input_shape=input_shape, padding='same'))
    model.add(layers.MaxPool2D())
    model.add(layers.Conv2D(filters=128, kernel_size=(2, 2), padding='same'))
    model.add(layers.MaxPool2D())
    model.add(layers.Conv2D(filters=9 * 10, kernel_size=(2, 2), padding='same'))
    units = np.prod(model.layers[-1].output_shape[1:])
    assert units % 9 == 0, 'inconsistent shape'
    model.add(layers.Reshape((9, units // 9)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.4))
    return model


def m3_drop_02():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(filters=64, kernel_size=(2, 2), input_shape=input_shape, padding='same'))
    model.add(layers.MaxPool2D())
    model.add(layers.Conv2D(filters=128, kernel_size=(2, 2), padding='same'))
    model.add(layers.MaxPool2D())
    model.add(layers.Conv2D(filters=9 * 10, kernel_size=(2, 2), padding='same'))
    units = np.prod(model.layers[-1].output_shape[1:])
    assert units % 9 == 0, 'inconsistent shape'
    model.add(layers.Reshape((9, units // 9)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.2))
    return model


def m3_drop_04():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(filters=64, kernel_size=(2, 2), input_shape=input_shape, padding='same'))
    model.add(layers.MaxPool2D())
    model.add(layers.Conv2D(filters=128, kernel_size=(2, 2), padding='same'))
    model.add(layers.MaxPool2D())
    model.add(layers.Conv2D(filters=9 * 10, kernel_size=(2, 2), padding='same'))
    units = np.prod(model.layers[-1].output_shape[1:])
    assert units % 9 == 0, 'inconsistent shape'
    model.add(layers.Reshape((9, units // 9)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.4))
    return model


def m3_drop_05():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(filters=64, kernel_size=(2, 2), input_shape=input_shape, padding='same'))
    model.add(layers.MaxPool2D())
    model.add(layers.Conv2D(filters=128, kernel_size=(2, 2), padding='same'))
    model.add(layers.MaxPool2D())
    model.add(layers.Conv2D(filters=9 * 10, kernel_size=(2, 2), padding='same'))
    units = np.prod(model.layers[-1].output_shape[1:])
    assert units % 9 == 0, 'inconsistent shape'
    model.add(layers.Reshape((9, units // 9)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    return model


select_nn_model = {
    'a_1': ('fs_adam_m1_conv_32', 'adam', m1_conv_32),
    'a_2': ('fs_adam_m1_conv_64', 'adam', m1_conv_64),
    'a_3': ('fs_adam_m1_conv_128', 'adam', m1_conv_128),
    'a_4': ('fs_adam_m1_conv_64_128_256', 'adam', m1_conv_64_128_256),
    'r_4': ('fs_adam_m1_conv_64_128_256', 'rmsprop', m1_conv_64_128_256),
    'a_5': ('fs_adam_m2_dense_32', 'adam', m2_dense_32),
    'a_6': ('fs_adam_m2_dense_64', 'adam', m2_dense_64),
    'r_6': ('fs_adam_m2_dense_64', 'rmsprop', m2_dense_64),
    'a_7': ('fs_adam_m2_dense_128', 'adam', m2_dense_128),
    'r_7': ('fs_adam_m2_dense_128', 'rmsprop', m2_dense_128),
    'a_8': ('fs_adam_m2_dense_256_drop_03', 'adam', m2_dense_256_drop_03),
    'a_9': ('fs_adam_m2_dense_256_drop_04', 'adam', m2_dense_256_drop_04),
    'a_10': ('fs_adam_m3_drop_02', 'adam', m3_drop_02),
    'a_11': ('fs_adam_m3_drop_04', 'adam', m3_drop_04),
    'a_12': ('fs_adam_m3_drop_05', 'adam', m3_drop_05)
    # '': ('',)
}

model_name, optimizer, model_fn = select_nn_model[model_name]
model = model_fn()
model.add(layers.Dense(len(alphabet), activation='softmax'))

print(model.summary())


# ## Define callbacks
class LRTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, log_dir, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': tf.keras.backend.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)


lr_log_cbk = LRTensorBoard(log_dir='/media/disk1/tensorboard/plate/lr/{}'.format(model_name))

tensorboard_cbk = tf.keras.callbacks.TensorBoard(
    log_dir='/media/disk1/tensorboard/plate/{}'.format(model_name))

checkpoint_cbk = tf.keras.callbacks.ModelCheckpoint(
    filepath=join('/media/disk1/weights', '%(model_name)s-{epoch:02d}.h5' % {'model_name': model_name}),
    monitor='val_loss',
    verbose=1, period=10)

reduce_lr_cbk = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=5,
    verbose=1,
    min_lr=0.00001)

# ## Fit model
model.compile(optimizer=optimizer,
              loss=plate_loss,
              metrics=[plate_acc])

model.fit(X, y,
          epochs=200, batch_size=32,
          validation_data=(X_test, y_test),
          callbacks=[tensorboard_cbk, checkpoint_cbk, reduce_lr_cbk, lr_log_cbk])

# ## Evaluation
print('train')
print(model.evaluate(X, y))

print('test')
print(model.evaluate(X_test, y_test))

# ## Save model

model.save('/media/disk1/weights/{}.h5'.format(model_name))
