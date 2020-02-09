#include <cmath>
#include <cassert>
#include <memory>
#include <iostream>
#include <random>
#include <chrono>

#include "nn.h"
#include "nn_nodes.h"

/*************************************************************************************************************************************/

static size_t arg_max(const vector& v)
{
	if (v.length == 0)
		return -1;

	float max = v[0];
	size_t i_max = 0;
	for (size_t i = 0; i < v.length; i++)
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

nn::nn(size_t input_size, std::vector<layer> layers)
{
	_layers.reserve(layers.size() * 2);
	_a.reserve(_layers.size() + 1);
	_dy.reserve(_layers.size() + 1);
	_dw.reserve(_layers.size());
	_db.reserve(_layers.size());

	_a.emplace_back(input_size);
	_dy.emplace_back(input_size);

	size_t i = 0;
	size_t layer_input = input_size;

	for (const auto& layer_info : layers)
	{
		i++;

		_layers.push_back(std::make_unique<layer_node>(layer_input, layer_info.size));
		_a.emplace_back(layer_info.size);
		_dy.emplace_back(layer_info.size);
		_dw.emplace_back(layer_info.size, layer_input);
		_db.emplace_back(layer_info.size);

		_layers.push_back(std::make_unique<activation_node>(layer_info.size, layer_info.actv, i == layers.size()));
		_a.emplace_back(layer_info.size);
		_dy.emplace_back(layer_info.size);
		_dw.emplace_back(0, 0);
		_db.emplace_back(0);

		layer_input = layer_info.size;
	}
}

void nn::train(std::vector<std::pair<vector, vector>> training_set, std::vector<std::pair<vector, vector>> testing_set, size_t epochs, float learning_rate, float regularization_term)
{
	const float k = learning_rate;
	const float r = 1 - (learning_rate * regularization_term / training_set.size());

	for (size_t ep = 0; ep < epochs; ep++)
	{
		std::cout << "epoch " << ep << ":" << std::endl;

		std::shuffle(training_set.begin(), training_set.end(), std::default_random_engine(ep));

		auto first = std::chrono::system_clock::now();
		auto last = first;
		size_t c = 0;
		for (const auto& p : training_set)
		{
			auto t = std::chrono::system_clock::now();
			if ((t - last) > std::chrono::seconds(1))
			{
				std::cout << "(" << c << "/" << training_set.size() << ") ";
				std::cout << std::round(1000.0f * (float)c / training_set.size()) / 10 << "%" << std::endl;
				last = t;
			}

			train_batch(p.first, p.second, k, r);

			c++;
		}

		double error = 0.0;
		size_t correct = 0;
		for (const auto& p : testing_set)
		{
			const auto& prediction = forward(p.first);

			for (size_t i = 0; i < prediction.length; i++)
				error += std::pow(prediction[i] - p.second[i], 2);

			if (arg_max(prediction) == arg_max(p.second))
				correct++;
		}
		error /= (2.0 * testing_set.size());

		std::cout << "mse: " << error << " | recognised: (" << correct << " / " << testing_set.size() << ")" << std::endl;
	}
}

void nn::train_batch(const vector& x, const vector& y, float k, float r)
{
	assert(_layers.back()->output_size() == y.length);
	assert(_layers.front()->input_size() == x.length);

	//forward prop
	for (size_t i = 0; i < x.length; i++)
		_a[0][i] = x[i];

	for (size_t i = 0; i < _layers.size(); i++)
		_layers[i]->forward(_a[i], _a[i + 1]);

	//backward prop
	loss_derivative(_a.back(), y, _dy.back());

	for (int i = _layers.size() - 1; i >= 0; i--)
		_layers[i]->backward(_a[i + 1], _a[i], _dy[i + 1], _dy[i], _dw[i], _db[i]);

	//gradient descent optimization
	for (size_t i = 0; i < _layers.size(); i++)
		_layers[i]->update_params(_dw[i], _db[i], k, r);
}

const vector& nn::forward(const vector& x)
{
	for (size_t i = 0; i < x.length; i++)
		_a[0][i] = x[i];

	for (size_t i = 0; i < _layers.size(); i++)
		_layers[i]->forward(_a[i], _a[i + 1]);

	return _a.back();
}

void nn::loss_derivative(const vector& y, const vector& t, vector& dy)
{
	assert(dy.length == t.length);
	assert(dy.length == y.length);

	for (size_t i = 0; i < dy.length; i++)
	{
		dy[i] = y[i] - t[i];
	}
}

/*************************************************************************************************************************************/
