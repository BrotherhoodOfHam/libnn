/*
	Simple fully connected layer
*/

#pragma once

#include "nodes.h"

namespace nn
{
	namespace nodes
	{
		class dense_layer : public node
		{
			tensor w, dw;
			tensor b, db;
			tensor y, dx;

		public:

			dense_layer(size_t input_size, size_t layer_size);

			const tensor& forward(const tensor& x) override;

			const tensor& backward(const tensor& x, const tensor& dy) override;

			void update_params(float k, float r) override;

			const tensor& weights() const { return w; }
			const tensor& biases() const { return b; }
		};
	}
}