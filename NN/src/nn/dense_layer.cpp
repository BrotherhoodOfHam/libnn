/*
	Simple fully connected layer
*/

#include <random>

#include "dense_layer.h"

using namespace nn;
using namespace nn::nodes;

/*************************************************************************************************************************************/

dense_layer::dense_layer(const tensor_shape& input_shape, size_t layer_size) :
	w(layer_size, input_shape.memory_size()), dw(layer_size, input_shape.memory_size()),
	b(layer_size), db(layer_size),
	y(layer_size), dx(input_shape.memory_size()),
	node(input_shape, layer_size)
{
	std::default_random_engine gen;
	std::normal_distribution<float> dist(0, 1);
	const float sqrtn = std::sqrt((float)input_shape.memory_size());

	for (size_t j = 0; j < layer_size; j++)
		for (size_t i = 0; i < input_shape.memory_size(); i++)
			w(j,i) = dist(gen) / sqrtn;

	for (size_t i = 0; i < layer_size; i++)
		b(i) = 0.0f;
}

const tensor& dense_layer::forward(const tensor& x)
{
	//for each row:
	//y = w.x + b
	for (size_t j = 0; j < w.shape(0); j++)
	{
		//z = w.x + b
		scalar z = 0.0f;
		for (size_t i = 0; i < w.shape(1); i++)
			z += x(i) * w(j,i);
		z += b(j);
		y(j) = z;
	}

	return y;
}

const tensor& dense_layer::backward(const tensor& x, const tensor& dy)
{
	// δ/δy = partial derivative of loss w.r.t to output
	// the goal is to find the derivatives w.r.t to parameters: δw, δb, δx

	// compute partial derivatives w.r.t weights and biases
	for (size_t j = 0; j < w.shape(0); j++)
	{
		// δb = δy
		db(j) = dy(j);
		// δw = δy * x 
		for (size_t i = 0; i < w.shape(1); i++)
			dw(j,i) = x(i) * dy(j);
	}

	// compute partial derivative w.r.t input x
	for (size_t i = 0; i < w.shape(1); i++)
	{
		// δx = w^T * δy
		scalar s = 0.0f;
		for (size_t j = 0; j < w.shape(0); j++)
			s += w(j,i) * dy(j);
		dx(i) = s;
	}

	return dx;
}

void dense_layer::update_params(float k, float r)
{
	// apply gradient descent
	for (size_t j = 0; j < w.shape(0); j++)
	{
		b(j) -= k * db(j);

		for (size_t i = 0; i < w.shape(1); i++)
			w(j,i) = (r * w(j, i)) - (k * dw(j, i));
	}
}

/*************************************************************************************************************************************/
