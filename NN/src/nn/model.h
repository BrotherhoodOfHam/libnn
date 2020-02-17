/*
	Neural Network Model
*/

#pragma once

#include "common.h"
#include "tensors.h"

/*************************************************************************************************************************************/

namespace nn
{
	namespace nodes
	{
		class node;
	}

	enum class activation
	{
		linear,
		sigmoid,
		tanh,
		relu,
		leaky_relu,
		softmax,
	};

	struct layer
	{
		size_t size;
		activation actv;
		float leakiness;
		float dropout;

		layer(size_t _data_size, activation _actv = activation::sigmoid, float _leakiness = 0.1f, float _dropout = 0.0f) :
			size(_data_size),
			actv(_actv),
			leakiness(_leakiness),
			dropout(_dropout)
		{}
	};

	class model
	{
		std::vector<std::unique_ptr<nodes::node>> _nodes;
		std::vector<std::reference_wrapper<const tensor>> _activations;
		float _learning_rate;

	public:

		model(size_t input_size, std::vector<layer> layers, float learning_rate);
		~model();

		model(const model&) = delete;

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

	private:

		nodes::node* input_node() { return _nodes.front().get(); }
		nodes::node* output_node() { return _nodes.back().get(); }

		const tensor& _forwards(const tensor& x, bool is_training = true);
		const tensor& _backwards(const tensor& dy, bool is_training = true);
		void _update();

		void loss_derivative(const tensor& y, const tensor& t, tensor& dy);
	};
}
