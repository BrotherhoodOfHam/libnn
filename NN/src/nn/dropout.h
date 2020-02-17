/*
	Dropout regularization node
*/

#pragma once

#include <random>

#include "nodes.h"

namespace nn
{
	namespace nodes
	{
		class dropout : public node
		{
			float _probability;
			std::bernoulli_distribution _distribution;
			tensor y, dx;
			tensor p;
			
		public:

			dropout(size_t input_size, float probability);

			const tensor& forward(const tensor& x) override;

			const tensor& backward(const tensor& x, const tensor& dy) override;

			void update_params(float k, float r) override {}
		};
	}
}
