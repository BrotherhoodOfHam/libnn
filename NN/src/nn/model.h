/*
	Neural Network Model
*/

#pragma once

#include <type_traits>

#include "common.h"
#include "tensors.h"
#include "nodes.h"

/*************************************************************************************************************************************/

namespace nn
{
	class model
	{
		std::vector<std::unique_ptr<nodes::node>> _nodes;
		std::vector<std::reference_wrapper<const buffer>> _activations;
		layout<2> _input_layout;
		layout<2> _output_layout;
		float _learning_rate;
		bool _compiled;

	public:

		model(size_t input_size, size_t max_batch_size, float learning_rate);
		~model();

		model(const model&) = delete;

		// add node
		template<typename node_type, typename = std::enable_if_t<std::is_convertible_v<node_type*, nodes::node*>>, typename ... args_type>
		void add(args_type&& ... args)
		{
			assert(!_compiled);
			nodes::node_shape shape = _nodes.empty() ? _input_layout.shape() : _nodes.back()->output_shape();
			_nodes.push_back(std::make_unique<node_type>(shape, std::forward<args_type>(args)...));
		}

		void compile();

		nodes::node_shape input_shape() const { return _input_layout.shape(); }
		nodes::node_shape output_shape() const { return _output_layout.shape(); }
		
		void train(
			const std::vector<buffer>& x_train,
			const std::vector<buffer>& y_train,
			const std::vector<buffer>& x_test,
			const std::vector<buffer>& y_test,
			size_t epochs
		);

		float train_batch(const buffer& x, const buffer& y);

		void train_from_gradient(const buffer& dy);

		const buffer& forward(const buffer& x);

		const buffer& forward_backwards(const buffer& x, const buffer& y);

		// serialization
		bool serialize(const std::string& filename);
		bool deserialize(const std::string& filename);

	private:

		nodes::node* input_node() { return _nodes.front().get(); }
		nodes::node* output_node() { return _nodes.back().get(); }

		const buffer& _forwards(const buffer& x, bool is_training = true);
		const buffer& _backwards(const buffer& dy, bool is_training = true);
		void _update();

		void loss_derivative(const buffer& y, const buffer& t, buffer& dy);
	};
}
