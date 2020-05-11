/*
	Python interop
*/

#pragma once

#include <pybind11/embed.h>

namespace py = pybind11;

#include "generator.h"

/*************************************************************************************************************************************/

class Python
{
	py::scoped_interpreter _scope;
	bool _ok = false;

	py::function get_nearest;

public:

	Python();

    void nearest_neighbour(const ImageOutput& img, ImageOutput& nearest);
};

/*************************************************************************************************************************************/
