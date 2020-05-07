import numpy as np
import keras
import os
from keras.models import load_model
from keras.datasets import mnist
from matplotlib import pyplot as plt
from PIL import Image

(x_train, _), (_, _) = mnist.load_data()

gan = load_model("gan/generator.hdf5")
dcgan = load_model("dcgan/generator.hdf5")

batch_size = 100
z = np.random.randn(batch_size,10)

def to_img(x):
    return ((x + 1) / 2) * 255

def dump_samples(images, folder):
    if not os.path.exists(folder):
        os.mkdir(folder)
    images = images.reshape(-1, 28, 28)
    for i, im in enumerate(images):
        Image.fromarray(im).resize((28*10,28*10), Image.NEAREST).convert("L").save(folder + "/i" + str(i) + ".png")

# GAN
dump_samples(to_img(gan.predict(z)), "dump_gan")
# DCGAN
dump_samples(to_img(dcgan.predict(z)), "dump_dcgan")
# Real
g = x_train[np.random.choice(x_train.shape[0], batch_size, replace=False)]
dump_samples(g, "dump_mnist")
