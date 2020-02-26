from keras.callbacks import ModelCheckpoint
from keras.optimizers import *
from keras.models import *
from network.discriminator import *
from network.generator import *
from network.gan import *
from network.registraion import *
from keras.utils import generic_utils as keras_generic_utils
from network.losses import *
import random
import cv2
import math
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf


def get_data(data_path):
    data_list = os.listdir(data_path)
    data_num = len(data_list)
    all_data = np.ndarray((data_num, data_shape[0], data_shape[1], data_shape[2], 1), "float32")
    for n in range(data_num):
        if n % 500 == 0:
            print('Done: {0}/{1} images'.format(n, data_num))
        vol = np.fromfile(data_path + "%05d.raw" % n, dtype="float")
        vol.shape = data_shape
        all_data[n, :, :, :, 0] = vol
    return all_data


if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    set_session(tf.Session(config=config))

    reg_model = reg_unet()
    gen_model = gen_unet()
    disc_model = disc_net()

    gan_model = gan(generator_model=gen_model, discriminator_model=disc_model)
    reg_gen_model = reg_gen(reg_model=reg_model, generator_model=gen_model)

    # ------------------------
    # Define Optimizers
    opt_discriminator = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    opt_dcgan = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    opt_reg = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # ---------------------
    # Compile DCGAN
    gen_model.trainable = True
    reg_model.trainable = False
    disc_model.trainable = False
    # loss = [cc3D(), gradientSimilarity(win=[5, 5, 5]), 'binary_crossentropy']
    loss = ['mae', gradientSimilarity(win=[1, 1, 1]), 'binary_crossentropy']
    loss_weights = [100, 0, 1]
    # loss_weights = [100, 3, 1]
    gan_model.compile(loss=loss, loss_weights=loss_weights, optimizer=opt_dcgan)

    # ---------------------
    # COMPILE DISCRIMINATOR
    disc_model.trainable = True
    disc_model.compile(loss='binary_crossentropy', optimizer=opt_discriminator)

    # ---------------------
    # COMPILE REG_MODEL
    disc_model.trainable = False
    reg_model.trainable = True
    gen_model.trainable = False
    loss_reg = [gradientSimilarity(win=[7, 7, 7]), gradientLoss('l2'), cc3D(win=[7, 7, 7])]
    # loss_weights_reg = [1, 0.5, 0.5]
    loss_weights_reg = [1, 0.7, 1.0]
    reg_gen_model.compile(loss=loss_reg, loss_weights=loss_weights_reg, optimizer=opt_reg)

    # reg_model.load_weights("models/model_at_epoch_1.h5")
    # gen_model.load_weights("models/Gen/model_at_epoch_0.h5")
    # disc_model.load_weights("models/Disc/model_at_epoch_0.h5")

    mr_data = get_data(mr_path)
    ct_data = get_data(ct_path)

    Y_true_batch = np.ones((batch_size, 1), dtype="float32")
    Y_fake_batch = np.zeros((batch_size, 1), dtype="float32")
    y_gen = np.ones((batch_size, 1), dtype="float32")
    zero_flow = np.zeros((batch_size, data_shape[0], data_shape[1], data_shape[2], 3), dtype="float32")

    for ep in range(epochs):
        print("epochs:" + str(ep))
        progbar = keras_generic_utils.Progbar(train_num)
        for mini_batch in range(0, train_num, batch_size):
            # -----------------------------------train discriminator-------------------------------------------
            disc_model.trainable = True
            idx_mr = np.random.choice(mr_data.shape[0], batch_size, replace=False)
            mr_batch = mr_data[idx_mr]
            ct_batch = ct_data[idx_mr]
            src_t, flow, ct_gen = reg_gen_model.predict([mr_batch, ct_batch])

            if random.randint(0, 1) == 0:
                X_disc_batch = np.concatenate((ct_batch, ct_gen), axis=0)
                Y_disc_batch = np.concatenate((Y_true_batch, Y_fake_batch), axis=0)
            else:
                X_disc_batch = np.concatenate((ct_gen, ct_batch), axis=0)
                Y_disc_batch = np.concatenate((Y_fake_batch, Y_true_batch), axis=0)

            disc_loss = disc_model.train_on_batch(X_disc_batch, Y_disc_batch)

            # --------------------------------------train generator-------------------------------------------
            disc_model.trainable = False
            reg_model.trainable = False
            gen_model.trainable = True
            idx_mr = np.random.choice(mr_data.shape[0], batch_size, replace=False)
            mr_batch = mr_data[idx_mr]
            ct_batch = ct_data[idx_mr]
            mr_t, flow, sim = reg_gen_model.predict([mr_batch, ct_batch])
            gen_loss = gan_model.train_on_batch(mr_t, [ct_batch, mr_t, y_gen])

            # --------------------------------------train reg-------------------------------------------------
            disc_model.trainable = False
            reg_model.trainable = True
            gen_model.trainable = False
            idx_mr = np.random.choice(mr_data.shape[0], batch_size, replace=False)
            mr_batch = mr_data[idx_mr]
            ct_batch = ct_data[idx_mr]
            reg_loss = reg_gen_model.train_on_batch([mr_batch, ct_batch], [ct_batch, zero_flow, ct_batch])


            # print losses
            D_log_loss = disc_loss
            mae_loss = gen_loss[1].tolist()
            ngf_gen_loss = gen_loss[2].tolist()
            gen_log_loss = gen_loss[3].tolist()
            ngf_reg_loss = reg_loss[1].tolist()
            flow_loss = reg_loss[2].tolist()
            cc_loss = reg_loss[3].tolist()

            if (mini_batch % 100 == 0):
                progbar.add(batch_size, values=[("Dis", D_log_loss),
                                                ("MAE", mae_loss),
                                                ("NGF_gen", ngf_gen_loss),
                                                ("FAKE", gen_log_loss),
                                                ("NGF_reg", ngf_reg_loss),
                                                ("FLOW", flow_loss),
                                                ("CC", cc_loss)])

        # save models
        reg_model.save('models/Reg/model_at_epoch_%d.h5' % ep)
        disc_model.save('models/Disc/model_at_epoch_%d.h5' % ep)
        gen_model.save('models/Gen/model_at_epoch_%d.h5' % ep)
