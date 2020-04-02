/*
	Simple fully connected layer
*/

#pragma once

#include "node.h"

namespace nn
{
	class dense_layer final : public parameterised_node
	{
		tensor<2> w, dw;
		tensor<1> b, db;
		tensor<2> y, dx;

	public:

		dense_layer(node_shape input_shape, size_t layer_size);

		extents input_shape() const override { return dx.shape(); }
		extents output_shape() const override { return y.shape(); }

		const buffer& forward(const buffer& x) override;
		const buffer& backward(const buffer& x, const buffer& dy) override;

		node_parameter get_w() const override { return node_parameter{ w.data(), dw.data() }; }
		node_parameter get_b() const override { return node_parameter{ b.data(), db.data() }; }
	};
}