/*
	Base header for nodes
*/

#pragma once

#include "tensors.h"

namespace nn
{
	namespace nodes
	{
		/*
			Node representing a single differentiable operation
		*/
		class node
		{
		private:

			tensor_shape _input_shape;
			tensor_shape _output_shape;
			bool         _is_training;

		public:

			node(const tensor_shape& input_shape, const tensor_shape& output_shape) :
				_input_shape(input_shape), _output_shape(output_shape)
			{}

			node(const node&) = delete;

			void set_training(bool is_training) { _is_training = is_training; }
			bool is_training() { return _is_training; }

			inline const tensor_shape& input_shape() const { return _input_shape; }
			inline const tensor_shape& output_shape() const { return _output_shape; }

			// forward propagate
			virtual const tensor& forward(const tensor& x) = 0;

			// back propagate the gradient
			virtual const tensor& backward(const tensor& x, const tensor& dy) = 0;

			// update parameters
			virtual void update_params(float k, float r) = 0;
		};
	}
}