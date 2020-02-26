from keras.layers import Flatten, Dense, Input, Reshape, merge, Lambda, GlobalAveragePooling3D
from keras.layers.convolutional import Conv3D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from parameters import *


def disc_net():
    stride = 2
    input_layer = Input(shape=(data_shape[0], data_shape[1], data_shape[2], 1))

    num_filters_start = 32

    nb_conv = 5
    filters_list = [num_filters_start * min(8, (2 ** i)) for i in range(nb_conv)]

    disc_out = Conv3D(num_filters_start, kernel_size=3, strides=stride,
                             padding='same', name='disc_conv_1')(input_layer)
    disc_out = LeakyReLU(alpha=0.2)(disc_out)

    for i, filter_size in enumerate(filters_list[1:]):
        name = 'disc_conv_{}'.format(i + 2)

        disc_out = Conv3D(filter_size, kernel_size=3, strides=stride,
                                 padding='same', name=name)(disc_out)
        disc_out = BatchNormalization(name=name + '_bn')(disc_out)
        disc_out = LeakyReLU(alpha=0.2)(disc_out)

    disc_out = GlobalAveragePooling3D()(disc_out)

    disc_out = Dense(1, activation='sigmoid', name="disc_dense")(disc_out)

    dis_model = Model(input=input_layer, output=disc_out, name="patch_gan")

    return dis_model



if __name__ == "__main__":
    disc_net()
