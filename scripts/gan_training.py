import os
import time
import json
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import keras
from keras.models import Sequential

def train_gan(G, D, data, epochs=300, batch_size=1, plot_images=True, outdir="./data"):

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    assert(D.input_shape == G.output_shape)

    D.trainable = False
    # setup gan
    gan = Sequential(name="GAN")
    gan.add(G)
    gan.add(D)
    gan.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(lr=0.0002, beta_1=0.5))

    assert(len(G.input_shape) == 2)

    z_size      = G.input_shape[1]
    data_count  = data.shape[0]
    data_shape  = data[0].shape

    input_shape = list(D.input_shape)
    input_shape[0] = -1
    data        = data.reshape(input_shape)
    batch_count = data_count // batch_size

    print("data_shape", data_shape)
    print("flat", data.shape)

    z_test = np.random.randn(100, z_size)

    # metrics
    g_cost = []
    d_cost = []
    epoch_time = []
    start = time.time()

    # training loop
    for e in range(epochs):
        print("epoch (%d / %d)" % (e+1, epochs))
        dcost = 0
        gcost = 0
        elapsed = 0
        # foreach batch

        for _ in tqdm(range(batch_count)):
            t = time.time()

            # randomly sample from x distribution
            xbatch = data[np.random.choice(data_count, batch_size, replace=False)]
            # randomly sample from z distribution
            zbatch = np.random.randn(batch_size, z_size)
            # generate a batch of fake samples
            gbatch = G.predict(zbatch)

            # train discriminator on real then fake
            D.trainable = True
            dcost += D.train_on_batch(xbatch, np.full(batch_size, 0.9)) # smooth label
            dcost += D.train_on_batch(gbatch, np.zeros(batch_size))

            zg = np.random.randn(batch_size, z_size)
            yg = np.ones(batch_size)

            # train generator
            D.trainable = False
            gcost += gan.train_on_batch(zg, yg)

            elapsed += (time.time() - t)

        gcost /= batch_count
        dcost /= batch_count

        print("G cost:", gcost, "D cost:", dcost)
        d_cost.append(dcost)
        g_cost.append(gcost)
        epoch_time.append(elapsed)

        G.save(os.path.join(outdir, "generator.hdf5"))
        D.save(os.path.join(outdir, "discriminator.hdf5"))
        
        # plot results in figure
        if plot_images:
            dim=(10, 10)
            figsize=(10, 10)
            gimages = G.predict(z_test)
            gimages = gimages.reshape((gimages.shape[0],) + data_shape)
            gimages = np.squeeze(gimages)
            plt.figure(figsize=figsize)
            for i in range(gimages.shape[0]):
                plt.subplot(dim[0], dim[1], i+1)
                plt.imshow(gimages[i], interpolation='nearest',cmap=None if gimages[i].ndim==3 else "gray")
                plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, ("g%d.png" % e)))
            plt.close()

        metrics = {
            "d_loss": d_cost,
            "g_loss": g_cost,
            "epoch_time": epoch_time,
            "total_elapsed": (time.time() - start),
            "epochs": epochs,
            "current_epoch": e
        }
        json.dump(metrics, open(os.path.join(outdir, "metrics.json"), "w"))
