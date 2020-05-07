import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU, Dropout, BatchNormalization, Conv2D, Conv2DTranspose, Flatten, Reshape
from keras.optimizers import Adam

from gan_training import train_gan

z_size = 10
image_size = 28*28

# load MNIST
(x_train, _), (_, _) = mnist.load_data()
x_train = ((x_train / 255) * 2) - 1
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

epochs = 300
batch_size = 100

G = Sequential(name="Generator")
G.add(Dense(256 * 7 * 7, input_dim=z_size))
G.add(LeakyReLU(alpha=0.2))
G.add(Reshape((7, 7, 256)))
# don't upsample
G.add(Conv2DTranspose(128, (5,5), strides=(1,1), padding="same", use_bias=False))
G.add(BatchNormalization())
G.add(LeakyReLU(0.2))
# upsample to 14x14
G.add(Conv2DTranspose(64, (5,5), strides=(2,2), padding="same", use_bias=False))
G.add(BatchNormalization())
G.add(LeakyReLU(0.2))
# upsample to 28x28
G.add(Conv2DTranspose(1, (5,5), strides=(2,2), padding="same", activation="tanh"))
G.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.0002, beta_1=0.5))

assert(G.output_shape == (None, 28, 28, 1))

D = Sequential(name="Discriminator")
D.add(Conv2D(64, (5,5), strides=(2,2), padding="same", input_shape=(28,28,1)))
D.add(LeakyReLU(0.2))
D.add(Dropout(0.3))
D.add(Conv2D(128, (5,5), strides=(2,2), padding="same"))
D.add(LeakyReLU(0.2))
D.add(Dropout(0.3))
D.add(Flatten())
D.add(Dense(1, activation="sigmoid"))
D.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.0002, beta_1=0.5))

train_gan(G, D, x_train, epochs, batch_size, outdir="dcgan")
