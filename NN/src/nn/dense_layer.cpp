/*
	Simple fully connected layer
*/

#include <random>

#include "dense_layer.h"

using namespace nn::nodes;

/*************************************************************************************************************************************/

dense_layer::dense_layer(size_t input_size, size_t layer_size) :
	w(layer_size, input_size),
	b(layer_size),
	parametric_node(input_size, layer_size)
{
	std::default_random_engine gen;
	std::normal_distribution<float> dist(0, 1);
	const float sqrtn = std::sqrt((float)input_size);

	for (size_t j = 0; j < layer_size; j++)
		for (size_t i = 0; i < input_size; i++)
			w[j][i] = dist(gen) / sqrtn;

	for (size_t i = 0; i < layer_size; i++)
		b[i] = 0.0f;
}

void dense_layer::forward(const vector& x, vector& y) const
{
	assert(y.length == w.rows);
	assert(x.length == w.cols);

	//for each row:
	//y = w.x + b
	for (size_t j = 0; j < w.rows; j++)
	{
		//z = w.x + b
		scalar z = 0.0f;
		for (size_t i = 0; i < w.cols; i++)
			z += x[i] * w[j][i];
		z += b[j];
		y[j] = z;
	}
}

void dense_layer::backward(const vector& y, const vector& x, const vector& dy, vector& dx, matrix& dw, vector& db) const
{
	assert(dw.rows == w.rows && dw.cols == w.cols);
	assert(db.length == b.length);
	assert(dx.length == input_size());
	assert(dy.length == output_size());
	assert(x.length == input_size());
	assert(y.length == output_size());

	// δ/δy = partial derivative of loss w.r.t to output
	// the goal is to find the derivatives w.r.t to parameters: δw, δb, δx

	// compute partial derivatives w.r.t weights and biases
	for (size_t j = 0; j < w.rows; j++)
	{
		// δb = δy
		db[j] = dy[j];
		// δw = δy * x 
		for (size_t i = 0; i < x.length; i++)
			dw[j][i] = x[i] * dy[j];
	}

	// compute partial derivative w.r.t input x
	for (size_t i = 0; i < w.cols; i++)
	{
		// δx = w^T * δy
		scalar s = 0.0f;
		for (size_t j = 0; j < w.rows; j++)
			s += w[j][i] * dy[j];
		dx[i] = s;
	}
}

void dense_layer::update_params(const matrix& dw, const vector& db, float k, float r)
{
	assert(dw.rows == w.rows && dw.cols == w.cols);
	assert(db.length == b.length);

	// apply gradient descent
	for (size_t j = 0; j < w.rows; j++)
	{
		b[j] -= k * db[j];

		for (size_t i = 0; i < w.cols; i++)
			w[j][i] = (r * w[j][i]) - (k * dw[j][i]);
	}
}

/*************************************************************************************************************************************/
