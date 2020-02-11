/*
	Neural Network Model
*/

#pragma once

#include "common.h"
#include "base.h"

/*************************************************************************************************************************************/

namespace nn
{
	namespace nodes
	{
		class node_base;
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

		layer(size_t _size, activation _actv = activation::sigmoid, float _leakiness = 0.1f) :
			size(_size),
			actv(_actv),
			leakiness(_leakiness)
		{}
	};

	class model
	{
		std::vector<std::unique_ptr<nodes::node_base>> _layers;
		std::vector<vector> _a;  // layer activations
		std::vector<vector> _dy; // layer gradient
		std::vector<matrix> _dw; // layer weight gradient
		std::vector<vector> _db; // layer bias gradient

	public:

		model(size_t input_size, std::vector<layer> layers);
		~model();

		model(const model&) = delete;

		size_t input_size() const;
		size_t output_size() const;

		void train(std::vector<std::pair<vector, vector>> training_set, std::vector<std::pair<vector, vector>> testing_set, size_t epochs, float learning_rate, float regularization_term);

		void train_batch(const vector& x, const vector& y, float k, float r);

		void train_from_gradient(const vector& dy, float k, float r);

		const vector& forward(const vector& x);

		const vector& forward_backwards(const vector& x, const vector& y);

	private:

		void _forwards(const vector& x);
		void _backwards(const vector& dy);
		void _update(float k, float r);

		void loss_derivative(const vector& y, const vector& t, vector& dy);
	};
}
