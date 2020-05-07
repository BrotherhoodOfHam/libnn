import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, ReLU, LeakyReLU, Dropout
from keras.optimizers import Adam

from gan_training import train_gan

z_size = 10
image_size = 28*28

# load MNIST
(x_train, _), (_, _) = mnist.load_data()
x_train = ((x_train / 255) * 2) - 1

epochs = 300
batch_size = 100

G = Sequential(name="Generator")
G.add(Dense(256, input_dim=z_size))
G.add(LeakyReLU(0.2))
G.add(Dense(512))
G.add(LeakyReLU(0.2))
G.add(Dense(1024))
G.add(LeakyReLU(0.2))
G.add(Dense(image_size, activation='tanh'))
G.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.0002, beta_1=0.5))

D = Sequential(name="Discriminator")
D.add(Dense(1024, input_dim=image_size))
D.add(LeakyReLU(0.2))
D.add(Dropout(0.3))
D.add(Dense(512))
D.add(LeakyReLU(0.2))
D.add(Dropout(0.3))
D.add(Dense(256))
D.add(LeakyReLU(0.2))
D.add(Dropout(0.3))
D.add(Dense(1, activation="sigmoid"))
D.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.0002, beta_1=0.5))

train_gan(G, D, x_train, epochs, batch_size, outdir="./gan")