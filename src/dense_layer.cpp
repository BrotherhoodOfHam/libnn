/*
	Simple fully connected layer
*/

#include <random>

#include "nn/dense_layer.h"

using namespace nn;

/*************************************************************************************************************************************/

dense_layer::dense_layer(node_shape input_shape, size_t layer_size) :
	w(layer_size, input_shape[1]), dw(layer_size, input_shape[1]),
	b(layer_size), db(layer_size),
	y(input_shape[0], layer_size), dx(input_shape[0], input_shape[1])
{
	assert(input_shape.size() == 2);
	
	std::default_random_engine gen;
	std::normal_distribution<float> dist(0, 1);
	const float sqrtn = std::sqrt((float)input_shape[1]);

	for (uint j = 0; j < w.shape(0); j++)
		for (uint i = 0; i < w.shape(1); i++)
			w[j][i] = dist(gen) / sqrtn;

	for (uint i = 0; i < b.shape(0); i++)
		b[i] = 0.0f;
}

const buffer& dense_layer::forward(const buffer& _x)
{
	auto x = _x.as_tensor(dx.layout());
	
	//for each row:
	//y = w.x + b
	foreach(y.layout(), [&](uint b, uint j) {
		//z = w.x + b
		scalar z = 0.0f;
		for (uint i = 0; i < w.shape(1); i++)
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

	const uint batch_size = x.shape(0);

	// compute partial derivatives w.r.t biases
	foreach(b.layout(), [&](uint i) {
		// δb = δy
		scalar sum = 0;
		for (uint b = 0; b < batch_size; b++)
			sum += dy[b][i];
		db[i] = sum;
	});

	// compute partial derivatives w.r.t weights
	foreach(w.layout(), [&](uint j, uint i) {
		// δw = δy * x (outer product)
		scalar sum = 0;
		for (uint b = 0; b < batch_size; b++)
			sum += dy[b][j] * x[b][i];
		dw[j][i] = sum;
	});

	// compute partial derivative w.r.t input x
	foreach(dx.layout(), [&](uint b, uint i) {
		// δx = w^T * δy
		scalar sum = 0;
		for (uint j = 0; j < w.shape(0); j++)
			sum += w[j][i] * dy[b][j];
		dx[b][i] = sum;
	});

	return dx.data();
}

void dense_layer::update_params(float k, float r)
{
	// apply gradient descent
	foreach(b.layout(), [&](uint i) {
		b[i] -= k * db[i];
	});

	foreach(w.layout(), [&](uint j, uint i) {
		//w[j][i] = (r * w[j][i]) - (k * dw[j][i]);
		w[j][i] -= k * dw[j][i];
	});
}

/*************************************************************************************************************************************/
