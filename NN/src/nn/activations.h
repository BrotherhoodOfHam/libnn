/*
	Activation functions
*/

#pragma once

#include "nodes.h"

namespace nn
{
	namespace activation
	{
		// Base node for activations
		class activation_node : public nodes::node
		{
		protected:

			tensor y, dx;

		public:


			activation_node(const tensor_shape& input) :
				y(input), dx(input),
				node(input, input)
			{}

			void update_params(float k, float r) override {}

		protected:

			template<class function_type>
			const tensor& activate(const tensor& x, const function_type& func)
			{
				for (size_t i = 0; i < x.data_size(); i++)
				{
					//y = f(x, [y])
					y.at_index(i) = func(x.at_index(i), y.at_index(i));
				}
				return y;
			}

			template<class function_type>
			const tensor& derivative(const tensor& x, const tensor& dy, const function_type& func)
			{
				for (size_t i = 0; i < x.data_size(); i++)
				{
					//dx = f'(x, y, dy, [dx])
					dx.at_index(i) = func(x.at_index(i), y.at_index(i), dy.at_index(i), dx.at_index(i));
				}
				return dx;
			}
		};

		// Linear function:
		//  f(x) = x
		class linear : public activation_node
		{
		public:

			using activation_node::activation_node;

			const tensor& forward(const tensor& x) override;
			const tensor& backward(const tensor& x, const tensor& dy) override;
		};

		// Sigmoid function:
		//  f(x) = 1 / (1 + e^-x)
		class sigmoid : public activation_node
		{
		public:

			using activation_node::activation_node;

			const tensor& forward(const tensor& x) override;
			const tensor& backward(const tensor& x, const tensor& dy) override;
		};

		// TanH function:
		//  f(x) = (e^x - e^-x) / (e^x + e^-x)
		class tanh : public activation_node
		{
		public:

			using activation_node::activation_node;

			const tensor& forward(const tensor& x) override;
			const tensor& backward(const tensor& x, const tensor& dy) override;
		};

		// Rectified Linear Unit function:
		//  f(x) = max(0, x)
		class relu : public activation_node
		{
		public:

			using activation_node::activation_node;

			const tensor& forward(const tensor& x) override;
			const tensor& backward(const tensor& x, const tensor& dy) override;
		};

		// Leaky Rectified Linear Unit function:
		//  f(x, k) = if x > 0 then x else k * x
		class leaky_relu : public activation_node
		{
			float _leakiness = 0.0f;

		public:

			leaky_relu(const tensor_shape& input_shape, float leakiness) :
				activation_node(input_shape),
				_leakiness(leakiness)
			{}

			const tensor& forward(const tensor& x) override;
			const tensor& backward(const tensor& x, const tensor& dy) override;
		};

		// Softmax function:
		//  f(x) = e^x / sum(e^x)
		class softmax : public activation_node
		{
		public:

			using activation_node::activation_node;
			
			const tensor& forward(const tensor& x) override;
			const tensor& backward(const tensor& x, const tensor& dy) override;
		};
	}
}
