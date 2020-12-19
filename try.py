import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.layers import Add
from tensorflow.python.keras.layers import PReLU, LeakyReLU
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, BatchNormalization
from tensorflow.python.keras.layers import Input, Lambda
from tensorflow.python.keras.models import Model


def net_single(model, lr):
    return net(model, tf.expand_dims(lr, axis=0))[0]


def net(model, lr_batch):
    lr_batch = tf.cast(lr_batch, tf.float32)
    sr_batch = model(lr_batch)
    sr_batch = tf.clip_by_value(sr_batch, 0, 255)
    sr_batch = tf.round(sr_batch)
    sr_batch = tf.cast(sr_batch, tf.uint8)
    return sr_batch

def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)

def normalizesol(x):
    return x / 255.0


def normalize_new(x):
    return x / 127.5 - 1


def denormalize_new(x):
    return (x + 1) * 127.5


#Please note parts of the implementation are borrowed and are not our own.

def upsample(x, n):
    out = Conv2D(n, kernel_size=3, padding='same')(x)
    out = Lambda(pixel_shuffle(scale=2))(out)
    out = PReLU(shared_axes=[1, 2])(out)
    return out

def res_block(x, n, momentum=0.8):
    out = Conv2D(n, kernel_size=3, padding='same')(x)
    out = BatchNormalization(momentum=momentum)(out)
    out = PReLU(shared_axes=[1, 2])(out)
    out = Conv2D(n, kernel_size=3, padding='same')(out)
    out = BatchNormalization(momentum=momentum)(out)
    out = Add()([x, out])
    return out


def sr_resnet(n=64, blocks=16):
    x = Input(shape=(None, None, 3))
    out = Lambda(normalizesol)(x)

    out = Conv2D(n, kernel_size=9, padding='same')(out)
    out = x = PReLU(shared_axes=[1, 2])(out)

    for i in range(blocks):
        out = res_block(out, n)

    out = Conv2D(n, kernel_size=3, padding='same')(out)
    out = BatchNormalization()(out)
    out = Add()([x, out])

    out = upsample(out, n * 4)
    out = upsample(out, n * 4)

    out = Conv2D(3, kernel_size=9, padding='same', activation='tanh')(out)
    out = Lambda(denormalize_new)(out)

    return Model(x, out)



generator = sr_resnet


def discriminator_block(y, n, strides=1, batchnorm=True, momentum=0.8):
    out = Conv2D(n, kernel_size=3, strides=strides, padding='same')(y)
    if batchnorm:
        out = BatchNormalization(momentum=momentum)(out)
    return LeakyReLU(alpha=0.2)(out)


def discriminator(n=64):
    y = Input(shape=(HR_SIZE, HR_SIZE, 3))
    out = Lambda(normalize_new)(y)

    out = discriminator_block(out, n, batchnorm=False)
    out = discriminator_block(out, n, strides=2)

    out = discriminator_block(out, n * 2)
    out = discriminator_block(out, n * 2, strides=2)

    out = discriminator_block(out, n * 4)
    out = discriminator_block(out, n * 4, strides=2)

    out = discriminator_block(out, n * 8)
    out = discriminator_block(out, n * 8, strides=2)

    out = Flatten()(out)

    out = Dense(1024)(out)
    out = LeakyReLU(alpha=0.2)(out)
    out = Dense(1, activation='sigmoid')(out)

    return Model(y, out)


def vgg_22():
    return _vgg(5)


def vgg_54():
    return _vgg(20)


def _vgg(output_layer):
    vgg = VGG19(input_shape=(None, None, 3), include_top=False)
    return Model(vgg.input, vgg.layers[output_layer].output)
