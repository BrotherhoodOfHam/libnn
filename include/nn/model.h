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
		std::vector<vector> _activations;
		uint _input_size;
		
	public:

		model(uint input_size) : _input_size(input_size) {}

		// add node
		template<typename node_type, typename = std::enable_if_t<std::is_convertible_v<node_type*, node*>>, typename ... args_type>
		void add(args_type&& ... args)
		{
			tensor_shape shape = _nodes.empty() ? tensor_shape(_input_size) : _nodes.back()->output_shape();
			_nodes.push_back(std::make_unique<node_type>(shape, std::forward<args_type>(args)...));
		}

		tensor_shape input_shape() const { return _nodes.front()->input_shape(); }
		tensor_shape output_shape() const { return _nodes.back()->output_shape(); }

		vector forward(context& dc, const vector& x);
		vector backward(context& dc, const vector& dy);

		auto begin() const { return _nodes.begin(); }
		auto end() const { return _nodes.end(); }
		uint length() const { return (uint)_nodes.size(); }

		// serialization
		bool serialize(const std::string& filename);
		bool deserialize(const std::string& filename);
	};


	class composite_model
	{
		model& _a;
		model& _b;

	public:

		composite_model(model& a, model& b) :
			_a(a), _b(b)
		{}

		tensor_shape input_shape() const { return _a.input_shape(); }
		tensor_shape output_shape() const { return _b.output_shape(); }

		vector forward(context& dc, const vector& x)
		{
			return _b.forward(dc, _a.forward(dc, x));
		}

		vector backward(context& dc, const vector& dy)
		{
			return _a.backward(dc, _b.backward(dc, dy));
		}
	};
}

/*************************************************************************************************************************************/
