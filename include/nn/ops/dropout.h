/*
	Dropout regularization node
*/

#pragma once

#include <random>

#include "node.h"

namespace nn
{
	class dropout final : public uniform_node
	{
		random_generator _rng;
		float _probability;
		vector _dropout;

	public:

		dropout(tensor_shape input_shape, float probability);

		vector forward(context& dc, const vector& x) override;
		vector backward(context& dc, const vector& x, const vector& dy) override;
	};
}
