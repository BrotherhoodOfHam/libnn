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

model::model(size_t input_size, std::vector<layer> layers, float learning_rate)
{
	_learning_rate = learning_rate;
	size_t layer_input = input_size;

	for (const auto& layer_desc : layers)
	{
		_nodes.push_back(std::make_unique<dense_layer>(layer_input, layer_desc.size));

		std::unique_ptr<activation_node> actv;

		switch (layer_desc.actv)
		{
		case activation::linear:
			actv = std::make_unique<linear_activation>(layer_desc.size);
			break;
		case activation::sigmoid:
			actv = std::make_unique<sigmoid_activation>(layer_desc.size);
			break;
		case activation::tanh:
			actv = std::make_unique<tanh_activation>(layer_desc.size);
			break;
		case activation::relu:
			actv = std::make_unique<relu_activation>(layer_desc.size);
			break;
		case activation::leaky_relu:
			actv = std::make_unique<leaky_relu_activation>(layer_desc.size, layer_desc.leakiness);
			break;
		case activation::softmax:
			actv = std::make_unique<softmax_activation>(layer_desc.size);
			break;
		}

		_nodes.push_back(std::move(actv));

		if (layer_desc.dropout > 0.0f)
			_nodes.push_back(std::make_unique<dropout>(layer_desc.size, layer_desc.dropout));

		layer_input = layer_desc.size;
	}

	_activations.reserve(_nodes.size() + 1);
}

model::~model()
{

}

tensor_shape model::input_size() const { return _nodes.front()->input_shape(); }
tensor_shape model::output_size() const { return _nodes.back()->output_shape(); }

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
			const auto& prediction = forward(x_test[i]);

			for (size_t i = 0; i < prediction.shape(0); i++)
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


float model::train_batch(const tensor& x, const tensor& y)
{
	tensor::check(output_node()->output_shape(), y.shape());
	tensor::check(input_node()->input_shape(), x.shape());

	//forward prop
	const auto& a = _forwards(x);

	tensor dy(output_node()->output_shape());

	//backward prop
	loss_derivative(a, y, dy);
	_backwards(dy);

	//optimize
	_update();

	float loss = 0.0f;
	for (size_t i = 0; i < dy.shape(0); i++)
		loss += dy(i) * dy(i);
	return loss;
}

const tensor& model::forward_backwards(const tensor& x, const tensor& t)
{
	tensor::check(output_node()->output_shape(), t.shape());
	tensor::check(input_node()->input_shape(), x.shape());

	//forward prop
	const tensor& y = _forwards(x);
	tensor dy(output_node()->output_shape());

	//backward prop
	loss_derivative(y, t, dy);
	return _backwards(dy);
}

void model::train_from_gradient(const tensor& dy)
{
	//backward prop
	_backwards(dy);
	//optimize
	_update();
}

const tensor& model::forward(const tensor& x)
{
	return _forwards(x, false);
}

/*************************************************************************************************************************************/

const tensor& model::_forwards(const tensor& x, bool is_training)
{
	_activations.clear();

	auto a = std::ref(x);
	_activations.push_back(a);

	for (auto& node : _nodes)
	{
		node->set_training(is_training);
		a = node->forward(a);
		_activations.push_back(a);
	}

	return a;
}

const tensor& model::_backwards(const tensor& dy, bool is_training)
{
	//_forwards must be called
	assert(_activations.size() > 0);

	auto d = std::ref(dy);

	for (int i = _nodes.size() - 1; i >= 0; i--)
	{
		auto node = _nodes[i].get();
		node->set_training(is_training);
		d = node->backward(_activations[i], d);
	}

	return d;
}

void model::_update()
{
	//apply optimization
	for (auto& node : _nodes)
		node->update_params(_learning_rate, 1);
}

void model::loss_derivative(const tensor& y, const tensor& t, tensor& dy)
{
	tensor::check(dy, t);
	tensor::check(dy, y);

	for (size_t i = 0; i < dy.shape(0); i++)
	{
		dy(i) = y(i) - t(i);
	}
}

/*************************************************************************************************************************************/
