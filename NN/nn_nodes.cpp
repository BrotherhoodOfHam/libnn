#include "nn_nodes.h"
#include <cmath>
#include <algorithm>
#include <random>

/*************************************************************************************************************************************/

void activation_node::forward(const vector& x, vector& y) const
{
	assert(x.length == y.length);

	switch (_type)
	{
	case activation::linear:
	{
		for (size_t i = 0; i < x.length; i++)
			y[i] = x[i];
		break;
	}
	case activation::relu:
	{
		for (size_t i = 0; i < x.length; i++)
			y[i] = std::max(x[i], 0.0f);
		break;
	}
	case activation::sigmoid:
	{
		for (size_t i = 0; i < x.length; i++)
			y[i] = 1.0f / (1.0f + std::exp(-x[i]));
		break;
	}
	case activation::softmax:
	{
		scalar sum = 0.0f;
		for (size_t i = 0; i < x.length; i++)
		{
			y[i] = std::exp(x[i]);
			sum += y[i];
		}
		for (size_t i = 0; i < x.length; i++)
			y[i] /= sum;
		break;
	}
	}
}

void activation_node::backward(const vector& y, const vector& x, const vector& dy, vector& dx, matrix& dw, vector& db) const
{
	assert(x.length == y.length);
	assert(x.length == dy.length);
	assert(x.length == dx.length);

	//workaround for when this is the last layer
	if (_isoutput)
	{
		for (size_t i = 0; i < x.length; i++)
			dx[i] = dy[i];
		return;
	}

	switch (_type)
	{
	case activation::linear:
	{
		for (size_t i = 0; i < x.length; i++)
			dx[i] = dy[i];
		break;
	}
	case activation::relu:
	{
		for (size_t i = 0; i < x.length; i++)
			if (x[i] > 0.0f)
				dx[i] = dy[i];
			else
				dx[i] = 0.0f;
		break;
	}
	case activation::sigmoid:
	{
		for (size_t i = 0; i < x.length; i++)
		{
			dx[i] = (y[i] * (1.0f - y[i])) * dy[i];
		}
		break;
	}
	case activation::softmax:
	{
		for (size_t j = 0; j < x.length; j++)
		{
			float sum = 0.0f;
			for (size_t i = 0; i < x.length; i++)
			{
				float dydz = (i == j)
					? y[i] * (1 - y[i])
					: -y[i] * y[j];

				sum += dy[i] * dydz;
			}
			dx[j] = sum;
		}
		break;
	}
	}
}

/*************************************************************************************************************************************/

layer_node::layer_node(size_t input_size, size_t layer_size) :
	w(layer_size, input_size),
	b(layer_size),
	node(input_size, layer_size)
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

void layer_node::forward(const vector& x, vector& y) const
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

void layer_node::backward(const vector& y, const vector& x, const vector& dy, vector& dx, matrix& dw, vector& db) const
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

void layer_node::update_params(const matrix& dw, const vector& db, float k, float r)
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
