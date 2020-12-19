from tensorflow.python.keras.layers import Add, BatchNormalization, Conv2D, Dense, Flatten, Input, LeakyReLU, PReLU, Lambda
from tensorflow.python.keras.models import Model
from vgg import vgg_22, vgg_54


import numpy as np
import tensorflow as tf


#Please note that not all of the implementation below is our own,
#some code has been borrowed with acknowledgement


#Helper Functions
def GAN_img(model, x):
    #Passes single img x through the model
    return GAN(model, tf.expand_dims(x, axis=0))[0]


def GAN(model, x):
    #Passes a batch of images through the model, clips the values to 255
    #Converts the resulting output to type uint8 and returns the super resolution batch
    
    x = tf.cast(x, tf.float32)
    out = tf.cast(tf.round(tf.clip_by_value(model(x), 0, 255)), tf.uint8)
    return out

def normalize_logistic(out):
    #Maps values from [0,255] to [0,1]
    return out / 255.0


def normalize_tanh(out):
    #Maps values from [0, 255] to [-1,1]
    return out / 127.5 - 1


def denormalize_tanh(out):
    #Maps values from [-1,1] to [0, 255]
    return (out + 1) * 127.5

#Generative Adversarial Network Architecture
def upsample(x, num_filters):
    #Deconvolves an image from MxN to 2Mx2N
    out = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    out = Lambda(pixel_shuffle(scale=2))(out)
    return PReLU(shared_axes=[1, 2])(out)


def res_block(x, num_filters, momentum=0.8):
    #Residual Block for Generator Network
    out = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    out = BatchNormalization(momentum=momentum)(out)
    out = PReLU(shared_axes=[1, 2])(out)
    out = Conv2D(num_filters, kernel_size=3, padding='same')(out)
    out = BatchNormalization(momentum=momentum)(out)
    out = Add()([x, out])
    return out


def sr_resnet(num_filters=64, num_res_blocks=16):
    #The resnet comprised of the residual blocks above, as prescribed by the
    #architecture in the original paper.

    #Is later set to be the generator after certain modifications to its parameters
    x = Input(shape=(None, None, 3))
    out = Lambda(normalize_logistic)(x)

    out = Conv2D(num_filters, kernel_size=9, padding='same')(out)
    out = x_1 = PReLU(shared_axes=[1, 2])(out)

    for _ in range(num_res_blocks):
        out = res_block(out, num_filters)

    out = Conv2D(num_filters, kernel_size=3, padding='same')(out)
    out = BatchNormalization()(out)
    out = Add()([x_1, out])

    out = upsample(out, num_filters * 4)
    out = upsample(out, num_filters * 4)

    out = Conv2D(3, kernel_size=9, padding='same', activation='tanh')(out)
    out = Lambda(denormalize_tanh)(out)

    return Model(x, out)


generator = sr_resnet


def pixel_shuffle(scale):
    return lambda out: tf.nn.depth_to_space(out, scale)



def discriminator_block(x, num_filters, strides=1, batchnorm=True, momentum=0.8):
    #A single discriminator block as recommended in architecture
    out = Conv2D(num_filters, kernel_size=3, strides=strides, padding='same')(x)
    if batchnorm:
        out = BatchNormalization(momentum=momentum)(out)
    return LeakyReLU(alpha=0.2)(out)


def discriminator(num_filters=64):
    #Cumulative disriminator network comprised of individual discriminator blocks with
    #varying number of kernels/convolution filters
    x = Input(shape=(HR_SIZE, HR_SIZE, 3))
    out = Lambda(normalize_tanh)(x)

    out = discriminator_block(out, num_filters, batchnorm=False)
    out = discriminator_block(out, num_filters, strides=2)

    out = discriminator_block(out, num_filters * 2)
    out = discriminator_block(out, num_filters * 2, strides=2)

    out = discriminator_block(out, num_filters * 4)
    out = discriminator_block(out, num_filters * 4, strides=2)

    out = discriminator_block(out, num_filters * 8)
    out = discriminator_block(out, num_filters * 8, strides=2)

    out = Flatten()(out)

    out = Dense(1024)(out)
    out = LeakyReLU(alpha=0.2)(out)
    out = Dense(1, activation='sigmoid')(out)

    return Model(x, out)

