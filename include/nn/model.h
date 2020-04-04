/*
	Neural Network Model
*/

#pragma once

#include "common.h"
#include "ops/node.h"

/*************************************************************************************************************************************/

namespace nn
{
	class model
	{
	protected:

		std::vector<std::unique_ptr<node>> _nodes;
		std::vector<std::reference_wrapper<const buffer>> _activations;
		layout<2> _input;

	public:

		model(uint input_size, uint max_batch_size) :
			_input(node_shape{ max_batch_size, input_size })
		{}

		// add node
		template<typename node_type, typename = std::enable_if_t<std::is_convertible_v<node_type*, node*>>, typename ... args_type>
		void add(args_type&& ... args)
		{
			auto shape = _nodes.empty() ? node_shape(_input) : _nodes.back()->output_shape();
			_nodes.push_back(std::make_unique<node_type>(shape, std::forward<args_type>(args)...));
		}

		node_shape input_shape() const { return _nodes.front()->input_shape(); }
		node_shape output_shape() const { return _nodes.back()->output_shape(); }

		const buffer& forward(const buffer& x, bool is_training = false);
		const buffer& backward(const buffer& dy, bool is_training = false);

		auto begin() const { return _nodes.begin(); }
		auto end() const { return _nodes.end(); }
		uint length() const { return (uint)_nodes.size(); }

		// serialization
		bool serialize(const std::string& filename);
		bool deserialize(const std::string& filename);
	};
}
