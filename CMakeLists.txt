#################################################################################################
#   libnn
#################################################################################################

CMAKE_MINIMUM_REQUIRED (VERSION 3.8)

project(libnn)

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
    DESTINATION "${PROJECT_BINARY_DIR}/mnist"
)

#################################################################################################
# libnn library
#################################################################################################

set(libnn_headers
    include/nn/ops/node.h
    include/nn/ops/activations.h
    include/nn/ops/dense.h
    include/nn/ops/dropout.h
    include/nn/common.h
    include/nn/tensors.h
    include/nn/sequence.h
	include/nn/training.h
    include/nn/model.h
)

set(libnn_sources
    src/ops/activations.cpp
    src/ops/dense.cpp
    src/ops/dropout.cpp
    src/common.cpp
    src/training.cpp
    src/model_serializer.cpp
    src/model.cpp
)

add_library(libnn ${libnn_headers} ${libnn_sources})
target_include_directories(libnn PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_compile_features(libnn PUBLIC cxx_std_17) # we need c++17 to use parallel algorithms

assign_source_groups(${libnn_headers} ${libnn_sources})

#################################################################################################

set(sample_sources
    sample/gan.h
    sample/gan.cpp
    sample/main.cpp
)

# sample program
add_executable(testing ${sample_sources})
target_link_libraries(testing PUBLIC libnn CImg MNIST)

#################################################################################################