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
#include "dropout.h"

using namespace nn;
using namespace nn::nodes;

/*************************************************************************************************************************************/

static size_t arg_max(const tensor& v)
{
	float max = v(0);
	size_t i_max = 0;
	for (size_t i = 0; i < v.shape(0); i++)
	{
		if (v(i) > max)
		{
			max = v(i);
			i_max = i;
		}
	}
	return i_max;
}

/*************************************************************************************************************************************/

model::model(size_t input_size, size_t max_batch_size, float learning_rate) :
	_input_shape(tensor_shape(max_batch_size, input_size)),
	_learning_rate(learning_rate),
	_compiled(false)
{}

model::~model() {}

tensor_shape model::input_shape() const { return _nodes.front()->input_shape(); }
tensor_shape model::output_shape() const { return _nodes.back()->output_shape(); }

void model::compile()
{
	_compiled = true;
	_activations.reserve(_nodes.size() + 1);
}

/*************************************************************************************************************************************/

void model::train(
	const std::vector<tensor>& x_train,
	const std::vector<tensor>& y_train,
	const std::vector<tensor>& x_test,
	const std::vector<tensor>& y_test,
	size_t epochs
)
{
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
		size_t c = 0;
		float training_loss = 0.0f;

		for (size_t i : indices)
		{
			auto t = std::chrono::system_clock::now();
			if ((t - last) > std::chrono::seconds(1))
			{
				std::cout << "(" << c << "/" << x_train.size() << ") ";
				std::cout << std::round(1000.0f * (float)c / x_train.size()) / 10 << "%" << "                            \r";
				last = t;
			}
			
			//training
			training_loss += train_batch(x_train[i], y_train[i]);

			c++;
		}

		std::cout << "(" << c << "/" << x_train.size() << ") 100%";
		std::cout << std::endl;

		training_loss /= x_train.size();
		double loss = 0.0;
		size_t correct = 0;

		for (size_t i = 0; i < x_test.size(); i++)
		{
			tensor prediction = forward(x_test[i]).reshape(tensor_shape(output_shape()[1]));

			for (size_t i = 0; i < prediction.shape(1); i++)
				loss += std::pow(prediction(i) - y_test[i](i), 2);

			if (arg_max(prediction) == arg_max(y_test[i]))
				correct++;
		}
		loss /= (2.0 * x_test.size());

		std::cout << "testing loss: " << loss
				<< " | training loss: " << training_loss
				<< " | accuracy: (" << correct << " / " << x_test.size() << ")" << std::endl;
	}
}


float model::train_batch(const tensor& _x, const tensor& _y)
{
	tensor x = _x.reshape(input_shape());
	tensor y = _y.reshape(output_shape());

	//forward prop
	const auto& a = _forwards(x);

	tensor dy(output_node()->output_shape());

	//backward prop
	loss_derivative(a, y, dy);
	_backwards(dy);

	//optimize
	_update();

	float loss = 0.0f;
	for (size_t b = 0; b < dy.shape(0); b++)
		for (size_t i = 0; i < dy.shape(1); i++)
			loss += dy(b, i) * dy(b,i);
	return loss;
}

const tensor& model::forward_backwards(const tensor& _x, const tensor& _t)
{
	tensor x = _x.reshape(input_shape());
	tensor t = _t.reshape(output_shape());
	
	//forward prop
	const tensor& y = _forwards(x);
	tensor dy(output_node()->output_shape());

	//backward prop
	loss_derivative(y, t, dy);
	return _backwards(dy);
}

void model::train_from_gradient(const tensor& _dy)
{
	tensor dy = _dy.reshape(output_shape());

	//backward prop
	_backwards(dy);
	//optimize
	_update();
}

const tensor& model::forward(const tensor& _x)
{
	tensor x = _x.reshape(input_shape());
	return _forwards(x, false);
}

/*************************************************************************************************************************************/

const tensor& model::_forwards(const tensor& x, bool is_training)
{
	assert(_compiled);
	
	auto a = std::cref(x);
	_activations.clear();
	_activations.push_back(a);

	for (auto& node : _nodes)
	{
		node->set_state(is_training);
		a = node->forward(a);
		_activations.push_back(a);
	}

	return a;
}

const tensor& model::_backwards(const tensor& dy, bool is_training)
{
	assert(_compiled);
	assert(tensor_shape::equals(dy.shape(), output_node()->output_shape()));
	//_forwards must be called
	assert(_activations.size() > 0);

	auto d = std::ref(dy);

	for (int i = _nodes.size() - 1; i >= 0; i--)
	{
		auto node = _nodes[i].get();
		node->set_state(is_training);
		d = node->backward(_activations[i], d);
	}

	return d;
}

void model::_update()
{
	assert(_compiled);

	//apply optimization
	for (auto& node : _nodes)
		node->update_params(_learning_rate, 1);
}

void model::loss_derivative(const tensor& y, const tensor& t, tensor& dy)
{
	for (size_t b = 0; b < dy.shape(0); b++)
	{
		for (size_t i = 0; i < dy.shape(1); i++)
		{
			dy(b,i) = y(b,i) - t(b,i);
		}
	}
}

/*************************************************************************************************************************************/
