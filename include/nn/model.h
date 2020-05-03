/*
	Neural Network Model
*/

#pragma once

#include "common.h"
#include "node/node.h"

/*************************************************************************************************************************************/

namespace nn
{
	/*
		A model is a differentiable program.

		It contains a sequence of operations and a set of learnable parameters.
	*/
	class model
	{
	protected:

		std::vector<std::shared_ptr<node>>	_nodes;
		std::vector<node_parameter>			_parameters;
		std::vector<batch>					_activations;
		dynamic_tensor_shape				_input_shape;

	public:

		template<typename ... args_type>
		explicit model(uint shape0, args_type ... shape) :
			_input_shape(std::initializer_list<uint>{ shape0, (uint)shape... })
		{}

		explicit model(tensor_shape shape) : _input_shape(shape) {}

		// Shape of input tensor
		tensor_shape input_shape()  const { return _input_shape; }
		// Shape of output tensor
		tensor_shape output_shape() const { return _nodes.empty() ? tensor_shape(_input_shape) : _nodes.back()->output_shape(); }

		// Forward and backward functions
		batch forward(scope& dc, const batch& x);
		batch backward(scope& dc, const batch& dy);

		batch execute(const batch& x);
		batch operator()(const batch& x) { return execute(x); }

		// Return the learnable parameters of this model
		const std::vector<node_parameter>& parameters() { return _parameters; }

		// Add a new node to this model
		template<typename node_type, typename = std::enable_if_t<std::is_convertible_v<node_type*, node*>>, typename ... args_type>
		model& add(args_type&& ... args)
		{
			_nodes.push_back(std::make_unique<node_type>(output_shape(), std::forward<args_type>(args)...));
			_nodes.back()->get_parameters(_parameters);
			return *this;
		}

		auto begin() const { return _nodes.cbegin(); }
		auto end() const { return _nodes.cend(); }
		uint length() const { return (uint)_nodes.size(); }

		// Serialize parameters to file
		bool serialize(const std::string& filename);
		// Deserialize parameters from file
		bool deserialize(const std::string& filename);

		// Return an immutable copy of this model, meaning it's parameters are hidden
		model immutable() const;

		// Return a new model so that the output of this model is piped into the next
		model compose(model& next) const;
	};
}

/*************************************************************************************************************************************/
