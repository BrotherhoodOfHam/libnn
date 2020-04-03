########################################################################################
#  Sample programs
########################################################################################

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/img")

add_executable(mnist_classifier
    mnist_classifier.cpp
)

add_executable(mnist_gan
    gan.h
    gan.cpp
    mnist_gan.cpp
)

target_link_libraries(mnist_classifier PUBLIC libnn CImg MNIST)
target_link_libraries(mnist_gan PUBLIC libnn CImg MNIST)
