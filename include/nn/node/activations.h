/*
	Activation functions
*/

#pragma once

#include "node.h"

namespace nn
{
	namespace activation
	{
		// Sigmoid function:
		//  f(x) = 1 / (1 + e^-x)
		class sigmoid : public uniform_node
		{
		public:

			using uniform_node::uniform_node;

			batch forward(scope& dc, const batch& x) override;
			batch backward(scope& dc, const batch& x, const batch& y, const batch& dy) override;
		};

		// TanH function:
		//  f(x) = (e^x - e^-x) / (e^x + e^-x)
		class tanh : public uniform_node
		{
		public:

			using uniform_node::uniform_node;

			batch forward(scope& dc, const batch& x) override;
			batch backward(scope& dc, const batch& x, const batch& y, const batch& dy) override;
		};

		// Rectified Linear Unit function:
		//  f(x) = max(0, x)
		class relu : public uniform_node
		{
		public:

			using uniform_node::uniform_node;

			batch forward(scope& dc, const batch& x) override;
			batch backward(scope& dc, const batch& x, const batch& y, const batch& dy) override;
		};

		// Leaky Rectified Linear Unit function:
		//  f(x, k) = if x > 0 then x else k * x
		class leaky_relu : public uniform_node
		{
			float _leakiness = 0.0f;

		public:

			using uniform_node::uniform_node;

			leaky_relu(tensor_shape input_shape, float leakiness) :
				uniform_node(input_shape),
				_leakiness(leakiness)
			{}

			batch forward(scope& dc, const batch& x) override;
			batch backward(scope& dc, const batch& x, const batch& y, const batch& dy) override;
		};

		// Softmax function:
		//  f(x) = e^x / sum(e^x)
		class softmax : public uniform_node
		{
		public:

			using uniform_node::uniform_node;

			batch forward(scope& dc, const batch& x) override;
			batch backward(scope& dc, const batch& x, const batch& y, const batch& dy) override;
		};
	}
}
