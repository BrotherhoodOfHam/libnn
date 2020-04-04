/*
	Dropout regularization node
*/

#include "nn/ops/dropout.h"

using namespace nn;

/*************************************************************************************************************************************/

dropout::dropout(node_shape input_shape, float probability) :
	y(input_shape.total()),
	dx(input_shape.total()),
	p(input_shape.total()),
	_probability(probability),
	_shape(input_shape)
{}

const buffer& dropout::forward(const buffer& _x)
{
	if (is_training())
	{
		auto x = _x.as_tensor(y.layout());

		auto rng = new_random_engine();

		dispatch(y.layout(), [&](uint i) {
			std::bernoulli_distribution dist(_probability);
			p[i] = dist(rng) ? 0.0f : 1.0f;
			y[i] = p[i] * x[i];
		});

		return y.data();
	}

	return _x;
}

const buffer& dropout::backward(const buffer& x, const buffer& _dy)
{
	if (is_training())
	{
		auto dy = _dy.as_tensor(y.layout());

		dispatch(y.layout(), [&](uint i) {
			dx[i] = p[i] * dy[i];
		});

		return dx.data();
	}

	return _dy;
}

/*************************************************************************************************************************************/
