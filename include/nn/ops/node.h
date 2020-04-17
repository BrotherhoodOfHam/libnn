/*
	Base header for nodes
*/

#pragma once

#include "../device.h"

namespace nn
{
	struct node_parameter
	{
		buffer p, dp;
		node_parameter(const buffer& _p, const buffer& _dp) : p(_p), dp(_dp) {}
	};

	/*
		Tensor variable
	*/
	template<uint rank>
	class variable
	{
		tensor_layout<rank>  _layout;
		buffer        _v;
		buffer        _dv;

	public:

		using tensor_slice = tensor<rank, scalar, internal::tensor_kind::is_proxy>;

		variable() = default;
		variable(const variable&) = default;

		template<typename ... args_t, typename = std::enable_if_t<rank == (sizeof...(args_t) + 1)>>
		variable(uint shape0, args_t ... shape) :
			_layout(shape0, shape...),
			_dv(_layout.total_size()),
			_v(_layout.total_size())
		{}

		inline uint size() const { return _layout.shape(0); }
		inline uint total_size() const { return _layout.total_size(); }
		inline uint shape(uint i) const { return _layout.shape(i); }
		inline tensor_shape shape() const { return _layout.shape(); }
		inline tensor_layout_view<rank> layout() const { return _layout; }

		node_parameter as_param() const { return node_parameter(_v, _dv); }

		inline tensor_slice v() const { return tensor_slice(_v.ptr(), layout()); }
		inline tensor_slice dv() const { return tensor_slice(_dv.ptr(), layout()); }
	};

	/*
		Node representing a single differentiable operation
	*/
	class node
	{
	public:

		// shapes
		virtual tensor_shape input_shape() const = 0;
		virtual tensor_shape output_shape() const = 0;

		// forward propagate
		virtual vector forward(context& dc, const vector& x) = 0;

		// back propagate the gradient
		virtual vector backward(context& dc, const vector& x, const vector& dy) = 0;
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

	/*
		Node representing operation that does not mutate the shape of it's input
	*/
	class uniform_node : public node
	{
		dynamic_tensor_shape _shape;

	public:

		uniform_node(tensor_shape shape) : _shape(shape) {}

		// shapes
		tensor_shape input_shape() const override { return _shape; }
		tensor_shape output_shape() const override { return _shape; }
	};
}