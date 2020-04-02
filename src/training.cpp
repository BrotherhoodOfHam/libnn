/*
	Neural Network Model
*/

#include <cmath>
#include <memory>
#include <iostream>
#include <random>
#include <chrono>

#include "nn/training.h"

using namespace nn;

/*************************************************************************************************************************************/

static size_t arg_max(const tensor_slice<1>& v)
{
	float max = v[0];
	uint i_max = 0;
	for (uint i = 0; i < v.shape(0); i++)
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

trainer::trainer(model& seq, float learning_rate) :
	_model(seq),
	_input_layout(seq.input_shape()),
	_output_layout(seq.output_shape()),
	_learning_rate(learning_rate)
{
	for (auto& node : _model)
	{
		auto p = dynamic_cast<parameterised_node*>(node.get());
		if (p != nullptr)
		{
			_parameters.push_back(p->get_w());
			_parameters.push_back(p->get_b());
		}
	}
}

trainer::~trainer() {}

/*************************************************************************************************************************************/

void trainer::train(
	const std::vector<data>&  x_train,
	const std::vector<label>& y_train,
	const std::vector<data>&  x_test,
	const std::vector<label>& y_test,
	size_t       epochs
)
{
	uint batch_size = _input_layout.shape(0);

	assert(x_train.size() % batch_size == 0);
	assert(x_test.size() % batch_size == 0);
	assert(x_train.size() == y_train.size());
	assert(x_test.size() == y_test.size());

	std::vector<size_t> indices(x_train.size());
	std::iota(indices.begin(), indices.end(), 0);

	tensor<2> input(_input_layout);
	tensor<2> output(_output_layout);

	zero_buffer(output.data());

	for (size_t ep = 0; ep < epochs; ep++)
	{
		std::cout << time_stamp << " epoch " << ep << ":" << std::endl;

		auto rng = new_random_engine();
		std::shuffle(indices.begin(), indices.end(), rng);

		auto first = std::chrono::system_clock::now();
		auto last = first;
		uint i_count = 0;
		uint i_iters = 0;
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

			uint batch_index = i_count % batch_size;

			if ((batch_index == 0) && (i_count > 0))
			{
				//training
				training_loss += train_batch(input.data(), output.data());

				zero_buffer(output.data());
			}

			update_tensor(input[batch_index], x_train[i_sample]);
			output[batch_index][y_train[i_sample]] = 1;

			i_count++;
			i_iters++;
		}

		std::cout << "(" << i_count << "/" << x_train.size() << ") 100%";
		std::cout << std::endl;

		training_loss /= x_train.size();
		double loss = 0.0;
		uint correct = 0;

		for (uint i_sample = 0; i_sample < x_test.size(); i_sample++)
		{
			uint batch_index = i_sample % batch_size;

			if ((batch_index == 0) && (i_sample > 0))
			{
				auto prediction = _model.forward(input.data()).as_tensor(output.layout());

				for (uint b = 0; b < output.shape(0); b++)
				{
					if (arg_max(prediction[b]) == arg_max(output[b]))
						correct++;

					for (uint i = 0; i < output.shape(1); i++)
						loss += std::pow(prediction[b][i] - output[b][i], 2);
				}

				zero_buffer(output.data());
			}

			update_tensor(input[batch_index], x_test[i_sample]);
			output[batch_index][y_test[i_sample]] = 1;
		}
		loss /= (2.0 * x_test.size());

		std::cout << "testing loss: " << loss
				<< " | training loss: " << training_loss
				<< " | accuracy: (" << correct << " / " << x_test.size() << ")" << std::endl;
	}
}

float trainer::train_batch(const buffer& x, const buffer& y)
{
	//forward prop
	const auto& a = _model.forward(x, true);

	auto dy = tensor(_output_layout);

	//backward prop
	loss_derivative(a, y, dy.data());
	_model.backward(dy.data(), true);

	//optimize
	update_parameters();

	float loss = 0.0f;
	for (uint b = 0; b < dy.shape(0); b++)
		for (uint i = 0; i < dy.shape(1); i++)
			loss += dy[b][i] * dy[b][i];
	return loss;
}

const buffer& trainer::forward_backwards(const buffer& x, const buffer& t)
{
	//forward prop
	const buffer& y = _model.forward(x);
	auto dy = tensor(_output_layout);

	//backward prop
	loss_derivative(y, t, dy.data());
	return _model.backward(dy.data(), true);
}

void trainer::train_from_gradient(const buffer& dy)
{
	//backward prop
	_model.backward(dy, true);
	//optimize
	update_parameters();
}

/*************************************************************************************************************************************/

void trainer::loss_derivative(const buffer& _y, const buffer& _t, buffer& _dy)
{
	auto y = _y.as_vector();
	auto dy = _dy.as_vector();
	auto t = _t.as_vector();

	foreach(dy.size(), [&](uint i) {
		dy[i] = y[i] - t[i];
	});
}

void trainer::update_parameters()
{
	//update using gradient descent
	for (auto& parameter : _parameters)
	{
		auto p  = parameter.p.as_vector();
		auto dp = parameter.dp.as_vector();

		// apply gradient descent
		foreach(parameter.p.size(), [&](uint i) {
			p[i] -= _learning_rate * dp[i];
		});
	}
}

/*************************************************************************************************************************************/
