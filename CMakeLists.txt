#################################################################################################
#   libnn
#################################################################################################

CMAKE_MINIMUM_REQUIRED (VERSION 3.9 FATAL_ERROR)

project(libnn LANGUAGES CUDA CXX)

option(BUILD_SAMPLES "Build sample programs" TRUE)
option(BUILD_DEMO "Build the demo visualization program" FALSE)
option(BUILD_PYTHON_BINDINGS "Build the python bindings" FALSE)

# vs project folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

include(InstallRequiredSystemLibraries)

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

# python script
#configure_file("${PROJECT_SOURCE_DIR}/scripts/libnn.py.in" "${PROJECT_SOURCE_DIR}/scripts/libnn.py")

#################################################################################################
# libnn library
#################################################################################################

set(libnn_headers
    include/nn/node/node.h
    include/nn/node/activations.h
    include/nn/node/dense.h
	include/nn/node/dropout.h
	include/nn/node/debug.h
    include/nn/common.h
    include/nn/tensors.h
	include/nn/training.h
	include/nn/optimizers.h
	include/nn/losses.h
	include/nn/model.h
	include/nn/device.h
)

set(libnn_sources
    src/node/activations.cu
    src/node/dense.cu
	src/node/dropout.cu
	src/node/debug.cpp

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
	PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
	PUBLIC "$<INSTALL_INTERFACE:include>"
	PRIVATE "${PROJECT_SOURCE_DIR}/src"
)
target_include_directories(libnn PRIVATE "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
#target_compile_features(libnn PUBLIC cxx_std_17) # we need c++17 to use parallel algorithms
target_compile_features(libnn PUBLIC cxx_std_14)
set_target_properties(libnn PROPERTIES
	CUDA_SEPARABLE_COMPILATION ON
	CUDA_RESOLVE_DEVICE_SYMBOLS ON
	POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(libnn PRIVATE cublas curand)
assign_source_groups(${libnn_headers} ${libnn_sources})

# install library
install(TARGETS libnn
	EXPORT libnn-targets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
# install library headers
install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/include/"
    DESTINATION "include"
)
# export cmake config
install(EXPORT libnn-targets
	FILE libnn-config.cmake
	DESTINATION lib/cmake/
)

#################################################################################################
# python binding
#################################################################################################

if (BUILD_PYTHON_BINDINGS)
	add_subdirectory(deps/pybind)

	pybind11_add_module(pylibnn
		src/python/module.cpp
	)
	target_link_libraries(pylibnn PRIVATE libnn)

	set_target_properties(pylibnn PROPERTIES
		LIBRARY_OUTPUT_NAME "libnn"
		SUFFIX ".pyd"
	)
	add_custom_command(TARGET pylibnn
		COMMAND "${CMAKE_COMMAND}" -E copy_if_different "$<TARGET_FILE:pylibnn>" "${PROJECT_SOURCE_DIR}/scripts"
	)
endif()

#################################################################################################
# Sample programs
#################################################################################################

if (BUILD_SAMPLES)
	add_subdirectory(samples)
endif()

#################################################################################################
# Demo program
#################################################################################################

if (BUILD_DEMO)
	add_subdirectory(demo)
endif()

#################################################################################################
