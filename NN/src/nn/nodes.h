/*
	Base header for nodes
*/

#pragma once

#include "tensors.h"

namespace nn
{
	namespace nodes
	{
		class differentiable_graph;

		/*
			Node representing a single differentiable operation
		*/
		class node
		{
		private:

			tensor_shape   _input_shape;
			tensor_shape   _output_shape;
			//const differentiable_graph& _graph;

			bool _is_training = false;

		public:

			bool is_training() const { return _is_training; }
			void set_state(bool is_training) { _is_training = is_training; }

			node(const tensor_shape& input_shape, const tensor_shape& output_shape) :
				_input_shape(input_shape), _output_shape(output_shape)//, _graph(graph)
			{}

			node(const node&) = delete;

			inline const tensor_shape& input_shape() const { return _input_shape; }
			inline const tensor_shape& output_shape() const { return _output_shape; }
			//inline const differentiable_graph& graph() const { return _graph; }

			// forward propagate
			virtual const tensor& forward(const tensor& x) = 0;

			// back propagate the gradient
			virtual const tensor& backward(const tensor& x, const tensor& dy) = 0;

			// update parameters
			virtual void update_params(float k, float r) = 0;
		};


		class differentiable_graph
		{
		protected:

			std::vector<std::unique_ptr<node>> _nodes;
			std::vector<std::reference_wrapper<const tensor>> _activations;

		public:

			float _learning_rate;

		};
	}
}