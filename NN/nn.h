/*
	Neural Network Model
*/

#pragma once

#include "base.h"

/*************************************************************************************************************************************/

enum class activation
{
	linear,
	sigmoid,
	relu,
	softmax
};

struct layer
{
	size_t size;
	activation actv;

	layer(size_t _size, activation _actv = activation::sigmoid) :
		size(_size),
		actv(_actv)
	{}
};

class nn
{
public:

	class node
	{
	private:

		size_t _input_size;
		size_t _output_size;

	public:

		node(size_t input_size, size_t output_size) :
			_input_size(input_size), _output_size(output_size)
		{}

		node(const node&) = delete;

		size_t input_size() const { return _input_size; }
		size_t output_size() const { return _output_size; }

		virtual void forward(const vector& x, vector& y) const = 0;

		virtual void backward(const vector& y, const vector& x, const vector& dy, vector& dx, matrix& dw, vector& db) const = 0;

		virtual void update_params(const matrix& dw, const vector& db, float r, float k) = 0;
	};

private:

	std::vector<std::unique_ptr<node>> _layers;
	std::vector<vector> _a;  // layer activations
	std::vector<vector> _dy; // layer gradient
	std::vector<matrix> _dw; // layer weight gradient
	std::vector<vector> _db; // layer bias gradient

public:

	nn(size_t input_size, std::vector<layer> layers);

	nn(const nn&) = delete;

	void train(std::vector<std::pair<vector, vector>> training_set, std::vector<std::pair<vector, vector>> testing_set, size_t epochs, float learning_rate, float regularization_term);

	void train_batch(const vector& x, const vector& y, float k, float r);

	const vector& forward(const vector& x);

private:

	void loss_derivative(const vector& y, const vector& t, vector& dy);
};

