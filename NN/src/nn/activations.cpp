/*
	Activation functions
*/

#include <cmath>
#include <algorithm>

#include "activations.h"

using namespace nn;

/*************************************************************************************************************************************/

const buffer& activation::linear::forward(const buffer& x)
{
	return x;
}

const buffer& activation::linear::backward(const buffer& x, const buffer& dy)
{
	return dy;
}

/*************************************************************************************************************************************/

const buffer& activation::sigmoid::forward(const buffer& x)
{
	return activate(x, [](scalar x, scalar y) {
		return 1.0f / (1.0f + std::exp(-x));
	});
}

const buffer& activation::sigmoid::backward(const buffer& x, const buffer& dy)
{
	// σ'(x) = σ(x) * (1 - σ(x))
	return derivative(x, dy, [](scalar x, scalar y, scalar dy, scalar dx) {
		return (y * (1.0f - y)) * dy;
	});
}

/*************************************************************************************************************************************/

const buffer& activation::tanh::forward(const buffer& x)
{
	return activate(x, [](scalar x, scalar y) {
		scalar a = std::exp(x);
		scalar b = 1.0f / a;
		return (a - b) / (a + b);
	});
}

const buffer& activation::tanh::backward(const buffer& x, const buffer& dy)
{
	// tanh'(x) = 1 - tanh(x)^2
	return derivative(x, dy, [](scalar x, scalar y, scalar dy, scalar dx) {
		return (1 - (y * y)) * dy;
	});
}

/*************************************************************************************************************************************/

const buffer& activation::relu::forward(const buffer& x)
{
	return activate(x, [](scalar x, scalar y) {
		return std::max(x, 0.0f);
	});
}

const buffer& activation::relu::backward(const buffer& x, const buffer& dy)
{
	return derivative(x, dy, [](scalar x, scalar y, scalar dy, scalar dx) {
		return (x > 0.0f) ? dy : 0;
	});
}

/*************************************************************************************************************************************/

const buffer& activation::leaky_relu::forward(const buffer& x)
{
	return activate(x, [=](scalar x, scalar y) {
		return (x > 0) ? x : _leakiness * x;
	});
}

const buffer& activation::leaky_relu::backward(const buffer& x, const buffer& dy)
{
	return derivative(x, dy, [=](scalar x, scalar y, scalar dy, scalar dx) {
		return ((x > 0) ? 1.0f : _leakiness) * dy;
	});
}

/*************************************************************************************************************************************/

const buffer& activation::softmax::forward(const buffer& x)
{
	scalar sum = 0.0f;

	activate(x, [&sum](scalar x, scalar y) {
		scalar a = std::exp(x);
		sum += a;
		return a;
	});
	return activate(x, [&sum](scalar x, scalar y) {
		return y / sum;
	});
}

const buffer& activation::softmax::backward(const buffer& x, const buffer& dy)
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
	return derivative(x, dy, [](scalar x, scalar y, scalar dy, scalar dx) {
		return dy;
	});
}

/*************************************************************************************************************************************/
