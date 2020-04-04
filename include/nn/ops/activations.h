/*
	Activation functions
*/

#pragma once

#include "node.h"

namespace nn
{
	namespace activation
	{
		// Base node for activations
		class activation_node : public node
		{
			dynamic_node_shape _shape;

		protected:

			tensor<1> y, dx;

		public:

			activation_node(node_shape input_shape) :
				_shape(input_shape),
				y(input_shape.total()), dx(y.layout())
			{}

			node_shape input_shape() const override { return _shape; }
			node_shape output_shape() const override { return _shape; }

		protected:

			template<class function_type>
			const buffer& activate(const buffer& _x, const function_type& func)
			{
				auto x = _x.as_vector();
				
				dispatch(x.size(), [&](uint i) {
					//y = f(x)
					y[i] = func(x[i]);
				});

				return y.data();
			}

			template<class function_type>
			const buffer& derivative(const buffer& _x, const buffer& _dy, const function_type& func)
			{
				auto x = _x.as_vector();
				auto dy = _dy.as_vector();

				dispatch(x.size(), [&](uint i) {
					//dx = f'(x, y, dy, [dx])
					dx[i] = func(x[i], y[i], dy[i], dx[i]);
				});

				return dx.data();
			}
		};

		// Sigmoid function:
		//  f(x) = 1 / (1 + e^-x)
		class sigmoid : public activation_node
		{
		public:

			using activation_node::activation_node;

			const buffer& forward(const buffer& x) override;
			const buffer& backward(const buffer& x, const buffer& dy) override;
		};

		// TanH function:
		//  f(x) = (e^x - e^-x) / (e^x + e^-x)
		class tanh : public activation_node
		{
		public:

			using activation_node::activation_node;

			const buffer& forward(const buffer& x) override;
			const buffer& backward(const buffer& x, const buffer& dy) override;
		};

		// Rectified Linear Unit function:
		//  f(x) = max(0, x)
		class relu : public activation_node
		{
		public:

			using activation_node::activation_node;

			const buffer& forward(const buffer& x) override;
			const buffer& backward(const buffer& x, const buffer& dy) override;
		};

		// Leaky Rectified Linear Unit function:
		//  f(x, k) = if x > 0 then x else k * x
		class leaky_relu : public activation_node
		{
			float _leakiness = 0.0f;

		public:

			leaky_relu(node_shape& input_shape, float leakiness) :
				activation_node(input_shape),
				_leakiness(leakiness)
			{}

			const buffer& forward(const buffer& x) override;
			const buffer& backward(const buffer& x, const buffer& dy) override;
		};

		// Softmax function:
		//  f(x) = e^x / sum(e^x)
		class softmax : public activation_node
		{
		public:

			using activation_node::activation_node;
			
			const buffer& forward(const buffer& x) override;
			const buffer& backward(const buffer& x, const buffer& dy) override;
		};
	}
}
