/*
	Debug layer
*/

#include "device/kernels.h"
#include "nn/ops/debug.h"

using namespace nn;

/*************************************************************************************************************************************/

debug_layer::debug_layer(tensor_shape shape, debug_flag flags) :
	uniform_node(shape), _flags(flags)
{}

vector debug_layer::forward(context& dc, const vector& x)
{
	if (_flags & debug_flag::print_forward)
	{
		auto _x = dc.to_batched(x, tensor_layout<1>(input_shape().total_size()));
		debug_print("forward:", _x);
	}
	return x;
}

vector debug_layer::backward(context& dc, const vector& x, const vector& dy)
{
	if (_flags & debug_flag::print_backward)
	{
		auto _dy = dc.to_batched(dy, tensor_layout<1>(output_shape().total_size()));
		debug_print("backward:", _dy);
	}
	return dy;
}

/*************************************************************************************************************************************/