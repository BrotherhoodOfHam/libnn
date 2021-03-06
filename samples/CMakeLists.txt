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

add_executable(xor_test xor.cpp)

########################################################################################

set_target_properties(mnist_classifier PROPERTIES FOLDER "samples")
set_target_properties(mnist_gan PROPERTIES FOLDER "samples")
set_target_properties(xor_test PROPERTIES FOLDER "samples")

target_link_libraries(mnist_classifier PUBLIC libnn MNIST)
target_link_libraries(mnist_gan PUBLIC libnn CImg MNIST)
target_link_libraries(xor_test PUBLIC libnn)

# for experimenting only
#add_executable(test_cuda test_cuda.cu)
#target_link_libraries(test_cuda PRIVATE cublas)
#set_target_properties(test_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#set_target_properties(test_cuda PROPERTIES FOLDER "samples")
