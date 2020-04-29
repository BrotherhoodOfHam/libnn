#################################################################################################
#   libnn
#################################################################################################

CMAKE_MINIMUM_REQUIRED (VERSION 3.9 FATAL_ERROR)

project(libnn LANGUAGES CXX CUDA)

# vs project folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#################################################################################################

function(assign_source_groups)
    
	foreach(_file ${ARGN})
		
		#Resolve absolute path
		get_filename_component(source_file "${_file}" ABSOLUTE)
		
		#Attempt to determine if the file is in the source or build tree
		string(FIND "${source_file}" "${CMAKE_CURRENT_SOURCE_DIR}" is_in_src)
		string(FIND "${source_file}" "${CMAKE_CURRENT_BINARY_DIR}" is_in_build)
		
		#If this file is in the build tree
		if(is_in_build EQUAL 0)
			file(RELATIVE_PATH source_file ${CMAKE_CURRENT_BINARY_DIR} ${source_file})
		#Otherwise if this file is in the source tree
		elseif(is_in_src EQUAL 0)
			file(RELATIVE_PATH source_file ${CMAKE_CURRENT_SOURCE_DIR} ${source_file})
		endif()
		
		#Get parent directory
		get_filename_component(source_dir "${source_file}" DIRECTORY)
		
		#Make sure we are using windows slashes
		#string(REPLACE "/" "\\" source_dir "${source_dir}")
		file(TO_NATIVE_PATH "${source_dir}" source_dir)
		
		#Debug print
		#message("[${is_in_src}||${is_in_build}]${source_file}")
		
		source_group("${source_dir}" FILES "${_file}")
		
	endforeach()
    
endfunction()

#################################################################################################
# External libraries
#################################################################################################

add_library(CImg INTERFACE)
target_include_directories(CImg INTERFACE ${PROJECT_SOURCE_DIR}/deps/CImg)

add_library(MNIST INTERFACE)
target_include_directories(MNIST INTERFACE ${PROJECT_SOURCE_DIR}/deps/mnist/include)

# copy mnist dataset to working directory
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/mnist")
file(COPY
    "${PROJECT_SOURCE_DIR}/deps/mnist/train-images-idx3-ubyte"
    "${PROJECT_SOURCE_DIR}/deps/mnist/train-labels-idx1-ubyte"
    "${PROJECT_SOURCE_DIR}/deps/mnist/t10k-images-idx3-ubyte"
    "${PROJECT_SOURCE_DIR}/deps/mnist/t10k-labels-idx1-ubyte"
    DESTINATION "${PROJECT_BINARY_DIR}/samples/mnist"
)

#################################################################################################
# libnn library
#################################################################################################

set(libnn_headers
    include/nn/ops/node.h
    include/nn/ops/activations.h
    include/nn/ops/dense.h
	include/nn/ops/dropout.h
	include/nn/ops/debug.h
    include/nn/common.h
    include/nn/tensors.h
	include/nn/training.h
	include/nn/optimizers.h
	include/nn/losses.h
	include/nn/model.h
	include/nn/device.h
)

set(libnn_sources
    src/ops/activations.cu
    src/ops/dense.cu
	src/ops/dropout.cu
	src/ops/debug.cpp

	src/device/gpu.h
	src/device/allocator.cpp
	src/device/device.cpp
	src/device/device_ops.cu
	src/device/random.cu

    src/common.cpp
    src/training.cpp
    src/model_serializer.cpp
    src/model.cpp
	src/losses.cu

    src/opt/adam.cu
    src/opt/sgd.cu
)

add_library(libnn ${libnn_headers} ${libnn_sources})
target_include_directories(libnn
	PUBLIC "${PROJECT_SOURCE_DIR}/include"
	PRIVATE "${PROJECT_SOURCE_DIR}/src"
)
#target_compile_features(libnn PUBLIC cxx_std_17) # we need c++17 to use parallel algorithms
target_compile_features(libnn PUBLIC cxx_std_14)
set_target_properties(libnn PROPERTIES
	CUDA_SEPARABLE_COMPILATION ON
	CUDA_RESOLVE_DEVICE_SYMBOLS ON
	POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(libnn PRIVATE cublas curand)

assign_source_groups(${libnn_headers} ${libnn_sources})

#################################################################################################

# Sample programs
add_subdirectory(samples)

#################################################################################################
