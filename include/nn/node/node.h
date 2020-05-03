/*
	Base header for nodes
*/

#pragma once

#include "../device.h"

namespace nn
{
	struct node_parameter
	{
		vector p, dp;
		node_parameter(const vector& _p, const vector& _dp) : p(_p), dp(_dp) {}
	};

	/*
		Tensor variable
	*/
	template<uint rank>
	class tensor_variable
	{
		tensor_layout<rank>  _layout;
		buffer				 _v;
		buffer				 _dv;

	public:

		using tensor_slice = tensor<rank, scalar, tensor_kind::is_slice>;

		tensor_variable() = default;
		tensor_variable(const tensor_variable&) = default;

		template<typename ... args_t, typename = std::enable_if_t<rank == (sizeof...(args_t) + 1)>>
		tensor_variable(uint shape0, args_t ... shape) :
			_layout(shape0, (uint)shape...),
			_dv(_layout.datasize()),
			_v(_layout.datasize())
		{}

		inline uint size() const { return _layout.datasize(); }
		inline uint shape(uint i) const { return _layout.shape(i); }
		inline tensor_shape shape() const { return _layout.shape(); }
		inline tensor_layout_view<rank> layout() const { return _layout; }

		operator node_parameter() const { return node_parameter(_v.as_vector(), _dv.as_vector()); }

		inline tensor_slice v() const { return tensor_slice(_v.ptr(), layout()); }
		inline tensor_slice dv() const { return tensor_slice(_dv.ptr(), layout()); }
	};

	/*
		input/output tensor batch.
	*/
	class batch : public tensor<2>
	{
	public:

		using tensor::tensor;

		template<uint n>
		tensor<n + 1> batch_reshape(const tensor_layout<n>& ly) const
		{
			return this->reshape(tensor_layout<n + 1>(this->shape(0), ly));
		}

		template<class ... args_t, uint n = 1 + sizeof...(args_t)>
		tensor<n + 1> batch_reshape(uint shape0, args_t ... shape) const
		{
			return this->batch_reshape(tensor_layout<n>(shape0, shape...));
		}
	};

	/*
		Node representing a single differentiable operation.

		An operation may expose learnable parameters.
	*/
	class node
	{
	public:

		// shapes
		virtual tensor_shape input_shape() const = 0;
		virtual tensor_shape output_shape() const = 0;

		// forward propagate
		virtual batch forward(scope& dc, const batch& x) = 0;

		// back propagate the gradient
		virtual batch backward(scope& dc, const batch& x, const batch& y, const batch& dy) = 0;

		// Enumerate any learnable parameters for this operation
		virtual void get_parameters(std::vector<node_parameter>& parameter_list) const { }
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