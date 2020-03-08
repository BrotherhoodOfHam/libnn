/*
	Dropout regularization node
*/

#pragma once

#include <random>

#include "nodes.h"

namespace nn
{
	class dropout final : public nodes::node
	{
		float _probability;
		tensor<1> y, dx;
		tensor<1> p;

		nodes::dynamic_node_shape _shape;

	public:

		dropout(nodes::node_shape input_shape, float probability);

		nodes::node_shape input_shape() const override { return _shape; }
		nodes::node_shape output_shape() const override { return _shape; }

		const buffer& forward(const buffer& x) override;
		const buffer& backward(const buffer& x, const buffer& dy) override;

		void update_params(float k, float r) override {}
	};
}
