/*
	Dropout regularization node
*/

#pragma once

#include "nodes.h"

namespace nn
{
	namespace nodes
	{
		class dropout : public node
		{
			float _probability;
			tensor v;
			
		public:

			dropout(size_t input_size, float probability);

			const tensor& forward(const tensor& x) override;

			const tensor& backward(const tensor& x, const tensor& dy) override;
		};
	}
}
