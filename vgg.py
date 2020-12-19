from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.models import Model

def vgg_22():
    return _vgg(5)


def vgg_54():
    return _vgg(20)

def _vgg(output_layer):
    vgg = VGG19(input_shape=(None, None, 3), include_top=False)
    return Model(vgg.input, vgg.layers[output_layer].output)
