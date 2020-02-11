/*
	Activation functions
*/

#pragma once

#include "nodes.h"

namespace nn
{
	namespace nodes
	{
		// Base node for activations
		class activation_node : public node
		{
		public:

			activation_node(size_t input_size) :
				node::node(input_size, input_size)
			{}
		};

		// Linear function:
		//  f(x) = x
		class linear_activation : public activation_node
		{
		public:

			using activation_node::activation_node;

			void forward(const vector& x, vector& y) const override;

			void backward(const vector& y, const vector& x, const vector& dy, vector& dx) const override;
		};

		// Sigmoid function:
		//  f(x) = 1 / (1 + e^-x)
		class sigmoid_activation : public activation_node
		{
		public:

			using activation_node::activation_node;

			void forward(const vector& x, vector& y) const override;

			void backward(const vector& y, const vector& x, const vector& dy, vector& dx) const override;
		};

		// TanH function:
		//  f(x) = (e^x - e^-x) / (e^x + e^-x)
		class tanh_activation : public activation_node
		{
		public:

			using activation_node::activation_node;

			void forward(const vector& x, vector& y) const override;

			void backward(const vector& y, const vector& x, const vector& dy, vector& dx) const override;
		};

		// Rectified Linear Unit function:
		//  f(x) = max(0, x)
		class relu_activation : public activation_node
		{
		public:

			using activation_node::activation_node;

			void forward(const vector& x, vector& y) const override;

			void backward(const vector& y, const vector& x, const vector& dy, vector& dx) const override;
		};

		// Leaky Rectified Linear Unit function:
		//  f(x, k) = if x > 0 then x else k * x
		class leaky_relu_activation : public activation_node
		{
			float _leakiness = 0.0f;

		public:

			leaky_relu_activation(size_t input_size, float leakiness) :
				activation_node(input_size),
				_leakiness(leakiness)
			{}

			void forward(const vector& x, vector& y) const override;

			void backward(const vector& y, const vector& x, const vector& dy, vector& dx) const override;
		};

		// Softmax function:
		//  f(x) = e^x / sum(e^x)
		class softmax_activation : public activation_node
		{
		public:

			using activation_node::activation_node;

			void forward(const vector& x, vector& y) const override;

			void backward(const vector& y, const vector& x, const vector& dy, vector& dx) const override;
		};
	}
}
