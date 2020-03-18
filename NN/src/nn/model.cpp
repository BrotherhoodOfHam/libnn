/*
	Neural Network Model
*/

#include <cmath>
#include <memory>
#include <iostream>
#include <random>
#include <chrono>

#include "model.h"
#include "activations.h"
#include "dense_layer.h"
#include "dropout.h"

using namespace nn;

/*************************************************************************************************************************************/

static size_t arg_max(const tensor_slice<1>& v)
{
	float max = v[0];
	size_t i_max = 0;
	for (size_t i = 0; i < v.shape(0); i++)
	{
		if (v[i] > max)
		{
			max = v[i];
			i_max = i;
		}
	}
	return i_max;
}

/*************************************************************************************************************************************/

model::model(size_t input_size, size_t max_batch_size, float learning_rate) :
	_input_layout(max_batch_size, input_size),
	_learning_rate(learning_rate),
	sequence(node_shape{ max_batch_size, input_size })
{}

model::~model() {}


/*************************************************************************************************************************************/

void model::train(
	const std::vector<buffer>& x_train,
	const std::vector<buffer>& y_train,
	const std::vector<buffer>& x_test,
	const std::vector<buffer>& y_test,
	size_t epochs
)
{
	assert(x_train.size() % _input_layout.shape()[0] == 0);
	assert(x_train.size() == y_train.size());
	assert(x_test.size() == y_test.size());

	std::vector<size_t> indices(x_train.size());
	std::iota(indices.begin(), indices.end(), 0);

	for (size_t ep = 0; ep < epochs; ep++)
	{
		std::cout << time_stamp << " epoch " << ep << ":" << std::endl;

		std::shuffle(indices.begin(), indices.end(), std::default_random_engine(ep));

		auto first = std::chrono::system_clock::now();
		auto last = first;
		size_t i_count = 0;
		size_t i_iters = 0;
		float training_loss = 0.0f;

		for (size_t i_sample : indices)
		{
			auto t = std::chrono::system_clock::now();
			if ((t - last) > std::chrono::seconds(1))
			{
				std::cout << "(" << i_count << "/" << x_train.size() << ") ";
				std::cout << std::round(1000.0f * (float)i_count / x_train.size()) / 10 << "% | "
					      << i_iters << "it/s                            \r";
				last = t;
				i_iters = 0;
			}
			
			//training
			training_loss += train_batch(x_train[i_sample], y_train[i_sample]);

			i_count++;
			i_iters++;
		}

		std::cout << "(" << i_count << "/" << x_train.size() << ") 100%";
		std::cout << std::endl;

		training_loss /= x_train.size();
		double loss = 0.0;
		size_t correct = 0;

		for (size_t i_sample = 0; i_sample < x_test.size(); i_sample++)
		{
			auto prediction = forward(x_test[i_sample]).as_vector();
			auto target = y_test[i_sample].as_vector();

			for (size_t i = 0; i < prediction.shape(1); i++)
				loss += std::pow(prediction[i] - target[i], 2);

			if (arg_max(prediction) == arg_max(target))
				correct++;
		}
		loss /= (2.0 * x_test.size());

		std::cout << "testing loss: " << loss
				<< " | training loss: " << training_loss
				<< " | accuracy: (" << correct << " / " << x_test.size() << ")" << std::endl;
	}
}


float model::train_batch(const buffer& x, const buffer& y)
{
	//forward prop
	const auto& a = forward(x);

	auto dy = tensor(layout<2>(output_shape()));

	//backward prop
	loss_derivative(a, y, dy.data());
	backward(dy.data());

	//optimize
	update_params(_learning_rate, 1);

	float loss = 0.0f;
	for (size_t b = 0; b < dy.shape(0); b++)
		for (size_t i = 0; i < dy.shape(1); i++)
			loss += dy[b][i] * dy[b][i];
	return loss;
}

const buffer& model::forward_backwards(const buffer& x, const buffer& t)
{
	//forward prop
	const buffer& y = forward(x);
	auto dy = tensor(layout<2>(output_shape()));

	//backward prop
	loss_derivative(y, t, dy.data());
	return backward(dy.data());
}

void model::train_from_gradient(const buffer& dy)
{
	//backward prop
	backward(dy);
	//optimize
	update_params(_learning_rate, 1);
}

/*************************************************************************************************************************************/

void model::loss_derivative(const buffer& _y, const buffer& _t, buffer& _dy)
{
	auto y = _y.as_vector();
	auto dy = _dy.as_vector();
	auto t = _t.as_vector();

	for_each(dy.size(), [&](uint i) {
		dy[i] = y[i] - t[i];
	});
}

/*************************************************************************************************************************************/
