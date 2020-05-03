/*
	Simple fully connected layer
*/

#pragma once

#include "node.h"

namespace nn
{
	class dense_layer final : public node
	{
		tensor_layout<1>   _input, _output;
		tensor_variable<2> _w;
		tensor_variable<1> _b;

	public:

		dense_layer(tensor_shape input_shape, uint layer_size);

		tensor_shape input_shape() const override { return _input.shape(); }
		tensor_shape output_shape() const override { return _output.shape(); }

		batch forward(scope& dc, const batch& x) override;
		batch backward(scope& dc, const batch& x, const batch& y, const batch& dy) override;

		void get_parameters(std::vector<node_parameter>& parameter_list) const override
		{
			parameter_list.push_back(_w);
			parameter_list.push_back(_b);
		}
	};
}