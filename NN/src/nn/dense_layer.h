/*
	Simple fully connected layer
*/

#pragma once

#include "nodes.h"

namespace nn
{
	namespace nodes
	{
		class dense_layer : public parametric_node
		{
			matrix w;
			vector b;

		public:

			dense_layer(size_t input_size, size_t layer_size);

			void forward(const vector& x, vector& y) const override;

			void backward(const vector& y, const vector& x, const vector& dy, vector& dx, matrix& dw, vector& db) const override;

			void update_params(const matrix& dw, const vector& db, float k, float r) override;
		};
	}
}