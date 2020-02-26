from keras.layers import Input, Lambda
from keras.models import Model
from parameters import *


def gan(generator_model, discriminator_model):
    src = Input(shape=data_shape + (1,))
    sim = generator_model(src)
    dcgan_output = discriminator_model(sim)
    dc_gan = Model(input=src, output=[sim, sim, dcgan_output])
    return dc_gan


def reg_gen(reg_model, generator_model):
    src = Input(shape=data_shape + (1,))
    tgt = Input(shape=data_shape + (1,))

    src_t, flow = reg_model([src, tgt])
    sim = generator_model(src_t)

    dc_gan = Model(input=[src, tgt], output=[src_t, flow, sim])
    return dc_gan
