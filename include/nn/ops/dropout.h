/*
	Dropout regularization node
*/

#pragma once

#include <random>

#include "node.h"

namespace nn
{
	class dropout final : public node
	{
		float _probability;
		tensor<1> y, dx;
		tensor<1> p;

		dynamic_node_shape _shape;

	public:

		dropout(node_shape input_shape, float probability);

		node_shape input_shape() const override { return _shape; }
		node_shape output_shape() const override { return _shape; }

		const buffer& forward(const buffer& x) override;
		const buffer& backward(const buffer& x, const buffer& dy) override;
	};
}
