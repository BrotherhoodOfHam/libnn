# Libnn

A simple Neural Network library implemented using CUDA. This project was developed for my final year project at university.

### Features:
* Dense layers
* Sigmoid, tanh, relu and softmax activation functions
* Dropout
* SGD and Adam optimizers
* Cross Entropy and Mean Squared Error loss functions

There are two main sample programs:
* MNIST [classifier](samples/mnist_classifier.cpp)
* MNIST [GAN](samples/mnist_gan.cpp)

Public headers are in [include/nn/](include/nn/).

Private source files are in [src](src/).

### Compiling

Requires [CMake](https://cmake.org/) and the [CUDA SDK](https://developer.nvidia.com/cuda-downloads). 
All other dependencies are included as git submodules in the [deps/](deps/) folder.

This program has only been tested on windows but will work on other operating systems in theory.

An example on windows:

```bash
git clone --recursive https://github.com/BrotherhoodOfHam/libnn.git
mkdir libnn/build
cd libnn/build
cmake .. -G "Visual Studio 16 2019" -A "x64"
cmake --build . --config Release
cd samples
./Release/mnist_classifier.exe
```