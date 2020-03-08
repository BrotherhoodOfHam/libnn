/*
	Simple fully connected layer
*/

#include <random>

#include "dense_layer.h"

using namespace nn;
using namespace nn::nodes;

/*************************************************************************************************************************************/

dense_layer::dense_layer(node_shape input_shape, size_t layer_size) :
	w(layer_size, input_shape[1]), dw(layer_size, input_shape[1]),
	b(layer_size), db(layer_size),
	y(input_shape[0], layer_size), dx(input_shape[0], input_shape[1])
{
	assert(input_shape.length() == 2);
	
	std::default_random_engine gen;
	std::normal_distribution<float> dist(0, 1);
	const float sqrtn = std::sqrt((float)input_shape[1]);

	for (size_t j = 0; j < w.shape(0); j++)
		for (size_t i = 0; i < w.shape(1); i++)
			w[j][i] = dist(gen) / sqrtn;

	for (size_t i = 0; i < b.shape(0); i++)
		b[i] = 0.0f;
}

const buffer& dense_layer::forward(const buffer& _x)
{
	auto x = _x.as_tensor(dx.layout());
	
	//for each row:
	//y = w.x + b
	for_each(y.layout(), [&](uint b, uint j) {
		//z = w.x + b
		scalar z = 0.0f;
		for (size_t i = 0; i < w.shape(1); i++)
			z += x[b][i] * w[j][i];
		z += x[b][j];
		y[b][j] = z;
	});

	return y.data();
}

const buffer& dense_layer::backward(const buffer& _x, const buffer& _dy)
{
	auto x = _x.as_tensor(dx.layout());
	auto dy = _dy.as_tensor(y.layout());

	// δ/δy = partial derivative of loss w.r.t to output
	// the goal is to find the derivatives w.r.t to parameters: δw, δb, δx

	const float batch_scale = 1.0f / x.shape(0);

	zero_buffer(dw.data());
	zero_buffer(db.data());

	// compute partial derivatives w.r.t weights and biases
	for_each(y.layout(), [&](uint b, uint j) {
		// δb = δy
		db[j] += dy[b][j] * batch_scale;
		// δw = δy * x 
		for (size_t i = 0; i < w.shape(1); i++)
			dw[j][i] += x[b][i] * dy[b][j] * batch_scale;
	});

	// compute partial derivative w.r.t input x
	for_each(dx.layout(), [&](uint b, uint i) {
		// δx = w^T * δy
		scalar s = 0.0f;
		for (size_t j = 0; j < w.shape(0); j++)
			s += w[j][i] * dy[b][j];
		dx[b][i] = s;
	});

	return dx.data();
}

void dense_layer::update_params(float k, float r)
{
	// apply gradient descent
	for (size_t j = 0; j < w.shape(0); j++)
	{
		b[j] -= k * db[j];

		for (size_t i = 0; i < w.shape(1); i++)
			w[j][i] = (r * w[j][i]) - (k * dw[j][i]);
	}
}

/*************************************************************************************************************************************/
