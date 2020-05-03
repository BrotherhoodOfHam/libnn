/*
	Debug layer
*/

#include "device/gpu.h"
#include "nn/node/debug.h"

using namespace nn;

/*************************************************************************************************************************************/

debug_layer::debug_layer(tensor_shape shape, debug_flag flags) :
	uniform_node(shape), _flags(flags)
{}

batch debug_layer::forward(scope& dc, const batch& x)
{
	if (_flags & debug_flag::print_forward)
	{
		debug_print("forward:", x);
	}
	return x;
}

batch debug_layer::backward(scope& dc, const batch& x, const batch& y, const batch& dy)
{
	if (_flags & debug_flag::print_backward)
	{
		debug_print("backward:", dy);
	}
	return dy;
}

/*************************************************************************************************************************************/
