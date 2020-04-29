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
		rng _rng;
		float _probability;
		vector _dropout;

	public:

		dropout(tensor_shape input_shape, float probability);

		vector forward(scope& dc, const vector& x) override;
		vector backward(scope& dc, const vector& x, const vector& dy) override;
	};
}
