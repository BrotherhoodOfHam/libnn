/*
	Neural Network Model
*/

#include <cmath>
#include <cassert>
#include <memory>
#include <iostream>
#include <random>
#include <chrono>

#include "model.h"
#include "activations.h"
#include "dense_layer.h"

using namespace nn;
using namespace nn::nodes;

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

model::model(size_t input_size, std::vector<layer> layers)
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

	for (const auto& layer_desc : layers)
	{
		i++;

		_layers.push_back(std::make_unique<dense_layer>(layer_input, layer_desc.size));
		_a.emplace_back(layer_desc.size);
		_dy.emplace_back(layer_desc.size);
		_dw.emplace_back(layer_desc.size, layer_input);
		_db.emplace_back(layer_desc.size);

		std::unique_ptr<node_base> act_ptr;

		switch (layer_desc.actv)
		{
		case activation::linear:
			act_ptr = std::make_unique<linear_activation>(layer_desc.size);
			break;
		case activation::sigmoid:
			act_ptr = std::make_unique<sigmoid_activation>(layer_desc.size);
			break;
		case activation::tanh:
			act_ptr = std::make_unique<tanh_activation>(layer_desc.size);
			break;
		case activation::relu:
			act_ptr = std::make_unique<relu_activation>(layer_desc.size);
			break;
		case activation::leaky_relu:
			act_ptr = std::make_unique<leaky_relu_activation>(layer_desc.size, layer_desc.leakiness);
			break;
		case activation::softmax:
			act_ptr = std::make_unique<softmax_activation>(layer_desc.size);
			break;
		}

		_layers.push_back(std::move(act_ptr));
		_a.emplace_back(layer_desc.size);
		_dy.emplace_back(layer_desc.size);
		_dw.emplace_back(0, 0);
		_db.emplace_back(0);

		layer_input = layer_desc.size;
	}
}

model::~model()
{

}

size_t model::input_size() const { return _layers.front()->input_size(); }
size_t model::output_size() const { return _layers.back()->output_size(); }

/*************************************************************************************************************************************/

void model::train(std::vector<std::pair<vector, vector>> training_set, std::vector<std::pair<vector, vector>> testing_set, size_t epochs, float learning_rate, float regularization_term)
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

void model::train_batch(const vector& x, const vector& y, float k, float r)
{
	assert(_layers.back()->output_size() == y.length);
	assert(_layers.front()->input_size() == x.length);

	//forward prop
	_forwards(x);

	//backward prop
	loss_derivative(_a.back(), y, _dy.back());
	_backwards(_dy.back());

	//optimize
	_update(k, r);
}

const vector& model::forward_backwards(const vector& x, const vector& y)
{
	assert(_layers.back()->output_size() == y.length);
	assert(_layers.front()->input_size() == x.length);

	//forward prop
	_forwards(x);

	//backward prop
	loss_derivative(_a.back(), y, _dy.back());
	_backwards(_dy.back());

	return _dy[0];
}

void model::train_from_gradient(const vector& dy, float k, float r)
{
	assert(_dy.back().length == dy.length);

	//backward prop
	_backwards(dy);
	//optimize
	_update(k, r);
}

const vector& model::forward(const vector& x)
{
	_forwards(x);
	return _a.back();
}

/*************************************************************************************************************************************/

void model::_forwards(const vector& x)
{
	assert(_a.front().length == x.length);

	//copy arguments
	for (size_t i = 0; i < x.length; i++)
		_a[0][i] = x[i];

	for (size_t i = 0; i < _layers.size(); i++)
		_layers[i]->forward(_a[i], _a[i + 1]);
}

void model::_backwards(const vector& dy)
{
	assert(_dy.back().length == dy.length);

	//copy arguments
	for (size_t i = 0; i < dy.length; i++)
		_dy.back()[i] = dy[i];

	for (int i = _layers.size() - 1; i >= 0; i--)
	{
		const node_base* layer = _layers[i].get();
		if (layer->type() == node_type::simple)
		{
			static_cast<const node*>(layer)->backward(_a[i + 1], _a[i], _dy[i + 1], _dy[i]);
		}
		else if (layer->type() == node_type::parametric)
		{
			static_cast<const parametric_node*>(layer)->backward(_a[i + 1], _a[i], _dy[i + 1], _dy[i], _dw[i], _db[i]);
		}
	}
}

void model::_update(float k, float r)
{
	//apply optimization
	for (size_t i = 0; i < _layers.size(); i++)
	{
		node_base* layer = _layers[i].get();

		if (layer->type() == node_type::parametric)
			static_cast<parametric_node*>(layer)->update_params(_dw[i], _db[i], k, r);
	}
}

void model::loss_derivative(const vector& y, const vector& t, vector& dy)
{
	assert(dy.length == t.length);
	assert(dy.length == y.length);

	for (size_t i = 0; i < dy.length; i++)
	{
		dy[i] = y[i] - t[i];
	}
}

/*************************************************************************************************************************************/
