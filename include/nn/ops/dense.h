/*
	Simple fully connected layer
*/

#pragma once

#include "node.h"

namespace nn
{
	class dense_layer final : public parameterised_node
	{
		tensor_layout<1> _input, _output;
		variable<2> _w;
		variable<1> _b;

	public:

		dense_layer(tensor_shape input_shape, uint layer_size);

		tensor_shape input_shape() const override { return _input.shape(); }
		tensor_shape output_shape() const override { return _output.shape(); }

		vector forward(context& dc, const vector& x) override;
		vector backward(context& dc, const vector& x, const vector& dy) override;

		node_parameter get_w() const override { return _w.as_param(); }
		node_parameter get_b() const override { return _b.as_param(); }
	};
}