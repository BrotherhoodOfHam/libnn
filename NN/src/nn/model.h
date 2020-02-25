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
	/*
	enum class activation
	{
		linear,
		sigmoid,
		tanh,
		relu,
		leaky_relu,
		softmax,
	};
	*/
	class base_model
	{
	protected:

		std::vector<std::unique_ptr<nodes::node>> _nodes;


	};

	class model
	{
		std::vector<std::unique_ptr<nodes::node>> _nodes;
		std::vector<std::reference_wrapper<const tensor>> _activations;
		tensor_shape _input_shape;
		float _learning_rate;
		bool _compiled;

	public:

		model(const tensor_shape& input_shape, size_t max_batch_size, float learning_rate);
		~model();

		model(const model&) = delete;

		// add node
		template<typename node_type, typename = std::enable_if_t<std::is_convertible_v<node_type*, nodes::node*>>, typename ... args_type>
		void add(args_type&& ... args)
		{
			const tensor_shape& shape = _nodes.empty() ? _input_shape : _nodes.back()->output_shape();
			_nodes.push_back(std::make_unique<node_type>(shape, std::forward<args_type>(args)...));
		}

		void compile();

		tensor_shape input_size() const;
		tensor_shape output_size() const;
		
		void train(
			const std::vector<tensor>& x_train,
			const std::vector<tensor>& y_train,
			const std::vector<tensor>& x_test,
			const std::vector<tensor>& y_test,
			size_t epochs
		);

		float train_batch(const tensor& x, const tensor& y);

		void train_from_gradient(const tensor& dy);

		const tensor& forward(const tensor& x);

		const tensor& forward_backwards(const tensor& x, const tensor& y);

		// serialization
		bool serialize(const std::string& filename);
		bool deserialize(const std::string& filename);

	private:

		nodes::node* input_node() { return _nodes.front().get(); }
		nodes::node* output_node() { return _nodes.back().get(); }

		const tensor& _forwards(const tensor& x, bool is_training = true);
		const tensor& _backwards(const tensor& dy, bool is_training = true);
		void _update();

		void loss_derivative(const tensor& y, const tensor& t, tensor& dy);
	};
}
