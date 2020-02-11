/*
	Activation functions
*/

#include <cmath>
#include <algorithm>

#include "activations.h"

using namespace nn;
using namespace nn::nodes;

/*************************************************************************************************************************************/

void linear_activation::forward(const vector& x, vector& y) const
{
	assert(x.length == y.length);

	for (size_t i = 0; i < x.length; i++)
		y[i] = x[i];
}

void linear_activation::backward(const vector& y, const vector& x, const vector& dy, vector& dx) const
{
	assert(x.length == y.length);
	assert(x.length == dy.length);
	assert(x.length == dx.length);

	for (size_t i = 0; i < x.length; i++)
		dx[i] = dy[i];
}

/*************************************************************************************************************************************/

void sigmoid_activation::forward(const vector& x, vector& y) const
{
	assert(x.length == y.length);

	for (size_t i = 0; i < x.length; i++)
		y[i] = 1.0f / (1.0f + std::exp(-x[i]));
}

void sigmoid_activation::backward(const vector& y, const vector& x, const vector& dy, vector& dx) const
{
	assert(x.length == y.length);
	assert(x.length == dy.length);
	assert(x.length == dx.length);

	// σ'(x) = σ(x) * (1 - σ(x))
	for (size_t i = 0; i < x.length; i++)
		dx[i] = (y[i] * (1.0f - y[i])) * dy[i];
}

/*************************************************************************************************************************************/

void tanh_activation::forward(const vector& x, vector& y) const
{
	assert(x.length == y.length);

	for (size_t i = 0; i < x.length; i++)
	{
		scalar a = std::exp(x[i]);
		scalar b = 1.0f / a;
		y[i] = (a - b) / (a + b);
	}
}

void tanh_activation::backward(const vector& y, const vector& x, const vector& dy, vector& dx) const
{
	assert(x.length == y.length);
	assert(x.length == dy.length);
	assert(x.length == dx.length);

	// tanh'(x) = 1 - tanh(x)^2
	for (size_t i = 0; i < x.length; i++)
		dx[i] = (1 - (y[i] * y[i])) * dy[i];
}

/*************************************************************************************************************************************/

void relu_activation::forward(const vector& x, vector& y) const
{
	assert(x.length == y.length);

	for (size_t i = 0; i < x.length; i++)
		y[i] = std::max(x[i], 0.0f);
}

void relu_activation::backward(const vector& y, const vector& x, const vector& dy, vector& dx) const
{
	assert(x.length == y.length);
	assert(x.length == dy.length);
	assert(x.length == dx.length);

	for (size_t i = 0; i < x.length; i++)
		dx[i] = (x[i] > 0.0f) ? dy[i] : 0;
}

/*************************************************************************************************************************************/

void leaky_relu_activation::forward(const vector& x, vector& y) const
{
	assert(x.length == y.length);

	for (size_t i = 0; i < x.length; i++)
		y[i] = (x[i] > 0) ? x[i] : _leakiness * x[i];
}

void leaky_relu_activation::backward(const vector& y, const vector& x, const vector& dy, vector& dx) const
{
	assert(x.length == y.length);
	assert(x.length == dy.length);
	assert(x.length == dx.length);

	for (size_t i = 0; i < x.length; i++)
		dx[i] = (x[i] > 0) ? dy[i] : _leakiness * dy[i];
}

/*************************************************************************************************************************************/

void softmax_activation::forward(const vector& x, vector& y) const
{
	assert(x.length == y.length);

	scalar sum = 0.0f;
	for (size_t i = 0; i < x.length; i++)
	{
		y[i] = std::exp(x[i]);
		sum += y[i];
	}
	for (size_t i = 0; i < x.length; i++)
		y[i] /= sum;
}

void softmax_activation::backward(const vector& y, const vector& x, const vector& dy, vector& dx) const
{
	/*
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
	*/

	// this is a workaround for when the softmax activation is the final node
	// when computing the p.d. the above method doesn't work with the cost function
	for (size_t i = 0; i < x.length; i++)
		dx[i] = dy[i];
}

/*************************************************************************************************************************************/
