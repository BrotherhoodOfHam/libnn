/*
	sequence node
*/

#include <type_traits>

#include "node.h"

#pragma once

namespace nn
{
	class sequence : public node
	{
	protected:

		std::vector<std::unique_ptr<node>> _nodes;
		std::vector<std::reference_wrapper<const buffer>> _activations;
		dynamic_node_shape _input;

	public:

		sequence(node_shape shape) :
			_input(shape)
		{}

		// add node
		template<typename node_type, typename = std::enable_if_t<std::is_convertible_v<node_type*, node*>>, typename ... args_type>
		void add(args_type&& ... args)
		{
			auto shape = _nodes.empty() ? node_shape(_input) : _nodes.back()->output_shape();
			_nodes.push_back(std::make_unique<node_type>(shape, std::forward<args_type>(args)...));
		}

		node_shape input_shape() const override { return _nodes.front()->input_shape(); }
		node_shape output_shape() const override { return _nodes.back()->output_shape(); }

		const buffer& forward(const buffer& x) override;

		const buffer& backward(const buffer& x, const buffer& dy) override;

		void update_params(float k, float r) override
		{
			for (auto& node : _nodes)
				node->update_params(k, r);
		}

		const buffer& backward(const buffer& dy) {
			return backward(buffer(), dy);
		}
	};
}
