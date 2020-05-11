#
#   Visualization demo of GAN generator
#

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set_property(GLOBAL PROPERTY AUTOGEN_SOURCE_GROUP "generated")

# Find Qt framework
find_package(Qt5 CONFIG COMPONENTS Widgets PATH_SUFFIXES "lib/cmake/Qt5" REQUIRED)

set(demo_src
	demo.cpp
	generator.h
	generator.cpp
	python_interop.h
	python_interop.cpp
)

add_executable(demo ${demo_src} models.qrc)
target_link_libraries(demo PRIVATE Qt5::Widgets libnn pybind11::embed)

source_group("src" FILES ${demo_src})

install(TARGETS demo LIBRARY DESTINATION "bin")

##############################################################################################################

#
#	Script adapted from this answer:
#	https://stackoverflow.com/questions/41193584/deploy-all-qt-dependencies-when-building
#
if (WIN32)
	
	if (TARGET Qt5::qmake AND NOT TARGET Qt5::windeployqt)
	
		# Set up imported target for convenience
		get_target_property(_qmake_executable Qt5::qmake IMPORTED_LOCATION)

		execute_process(
			COMMAND "${_qmake_executable}" -query QT_INSTALL_PREFIX
			RESULT_VARIABLE return_code
			OUTPUT_VARIABLE qt5_install_prefix
			OUTPUT_STRIP_TRAILING_WHITESPACE
		)
		
		set(_imported_location "${qt5_install_prefix}/bin/windeployqt.exe")

		if(EXISTS ${_imported_location})
			add_executable(Qt5::windeployqt IMPORTED)

			set_target_properties(Qt5::windeployqt PROPERTIES
				IMPORTED_LOCATION ${_imported_location}
			)
		else()
			message(WARNING "Unable to find windeployqt.exe")
		endif()
	endif()
	
	if (TARGET Qt5::windeployqt)
		# execute windeployqt in a tmp directory
		add_custom_target(DEPLOY ALL
			COMMAND ${CMAKE_COMMAND} -E remove_directory "${CMAKE_CURRENT_BINARY_DIR}/windeployqt"
			COMMAND set PATH=%PATH%$<SEMICOLON>${qt5_install_prefix}/bin
			COMMAND Qt5::windeployqt
					$<$<CONFIG:Debug>:"--debug">
					$<$<CONFIG:Release>:"--release">
					#--release
					--no-translations
					--no-system-d3d-compiler
					--dir "${CMAKE_CURRENT_BINARY_DIR}/windeployqt"
					"$<TARGET_FILE:demo>"
		)

		# copy deployment directory during installation
		install(
			DIRECTORY
			"${CMAKE_CURRENT_BINARY_DIR}/windeployqt/"
			DESTINATION bin
		)
	endif()

	# Copy Qt DLL's to the same location as the application executable
	foreach(_target
		Qt5::Widgets
		Qt5::Gui
		Qt5::Core
	)
		add_custom_command(
			TARGET demo POST_BUILD
			COMMENT "Copying ${_target}"
			COMMAND ${CMAKE_COMMAND} -E copy_if_different
				$<TARGET_FILE:${_target}>
				$<TARGET_FILE_DIR:demo>
		)
	endforeach()
endif()

##############################################################################################################
