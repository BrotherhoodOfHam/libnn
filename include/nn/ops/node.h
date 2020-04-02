/*
	Base header for nodes
*/

#pragma once

#include "../tensors.h"

namespace nn
{
	using node_shape = extents;
	using dynamic_node_shape = std::vector<uint>;

	/*
		Node representing a single differentiable operation
	*/
	class node
	{
		bool _is_training = false;

	public:

		node() = default;
		node(const node&) = delete;

		// state
		bool is_training() const { return _is_training; }
		void set_state(bool is_training) { _is_training = is_training; }

		// shapes
		virtual node_shape input_shape() const = 0;
		virtual node_shape output_shape() const = 0;

		// forward propagate
		virtual const buffer& forward(const buffer& x) = 0;

		// back propagate the gradient
		virtual const buffer& backward(const buffer& x, const buffer& dy) = 0;
	};

	struct node_parameter
	{
		buffer p, dp;
	};

	/*
		Node representing a differentiable operation with 2 learnable parameters
	*/
	class parameterised_node : public node
	{
	public:

		virtual node_parameter get_w() const = 0;
		virtual node_parameter get_b() const = 0;
	};
}