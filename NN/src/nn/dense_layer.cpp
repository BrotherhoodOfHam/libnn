/*
	Simple fully connected layer
*/

#include <random>

#include "dense_layer.h"

using namespace nn;
using namespace nn::nodes;

/*************************************************************************************************************************************/

dense_layer::dense_layer(const tensor_shape& input_shape, size_t layer_size) :
	w(layer_size, input_shape[1]), dw(layer_size, input_shape[1]),
	b(layer_size), db(layer_size),
	y(input_shape[0], layer_size), dx(input_shape[0], input_shape[1]),
	node(input_shape, { input_shape[0], layer_size })
{
	assert(input_shape.length() == 2);
	
	std::default_random_engine gen;
	std::normal_distribution<float> dist(0, 1);
	const float sqrtn = std::sqrt((float)input_shape[1]);

	for (size_t j = 0; j < w.shape(0); j++)
		for (size_t i = 0; i < w.shape(1); i++)
			w(j,i) = dist(gen) / sqrtn;

	for (size_t i = 0; i < b.shape(0); i++)
		b(i) = 0.0f;
}

const tensor& dense_layer::forward(const tensor& x)
{
	//for each row:
	//y = w.x + b
	for (size_t b = 0; b < x.shape(0); b++)
	{
		for (size_t j = 0; j < w.shape(0); j++)
		{
			//z = w.x + b
			scalar z = 0.0f;
			for (size_t i = 0; i < w.shape(1); i++)
				z += x(b, i) * w(j, i);
			z += x(b,j);
			y(b, j) = z;
		}
	}

	return y;
}

const tensor& dense_layer::backward(const tensor& x, const tensor& dy)
{
	// δ/δy = partial derivative of loss w.r.t to output
	// the goal is to find the derivatives w.r.t to parameters: δw, δb, δx

	const float batch_scale = 1.0f / x.shape(0);

	zero_gradients();

	// compute partial derivatives w.r.t weights and biases
	for (size_t b = 0; b < x.shape(0); b++)
	{
		for (size_t j = 0; j < w.shape(0); j++)
		{
			// δb = δy
			db(j) += dy(b,j) * batch_scale;
			// δw = δy * x 
			for (size_t i = 0; i < w.shape(1); i++)
				dw(j, i) += x(b,i) * dy(b,j) * batch_scale;
		}
	}

	// compute partial derivative w.r.t input x
	for (size_t b = 0; b < x.shape(0); b++)
	{
		for (size_t i = 0; i < w.shape(1); i++)
		{
			// δx = w^T * δy
			scalar s = 0.0f;
			for (size_t j = 0; j < w.shape(0); j++)
				s += w(j,i) * dy(b,j);
			dx(b,i) = s;
		}
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

void dense_layer::zero_gradients()
{
	for (size_t j = 0; j < dw.shape(0); j++)
	{
		db(j) = 0.0f;
		for (size_t i = 0; i < dw.shape(1); i++)
			dw(j, i) = 0.0f;
	}
}

/*************************************************************************************************************************************/
