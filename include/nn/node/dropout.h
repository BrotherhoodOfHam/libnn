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

		batch forward(scope& dc, const batch& x) override;
		batch backward(scope& dc, const batch& x, const batch& y, const batch& dy) override;
	};
}
