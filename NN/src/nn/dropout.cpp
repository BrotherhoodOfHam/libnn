/*
	Dropout regularization node
*/

#include "dropout.h"

using namespace nn;
using namespace nn::nodes;

/*************************************************************************************************************************************/

dropout::dropout(size_t input_size, float probability) :
	y(input_size),
	dx(input_size),
	p(input_size),
	_probability(probability),
	_distribution(probability),
	node(input_size, input_size)
{
}

const tensor& dropout::forward(const tensor& x)
{
	std::default_random_engine rng;

	if (is_training())
	{
		for (size_t i = 0; i < x.memory_size(); i++)
		{
			p(i) = _distribution(rng) ? 0.0f : 1.0f;
			y.at_index(i) = p(i) * x.at_index(i);
		}
	}
	return x;
}

const tensor& dropout::backward(const tensor& x, const tensor& dy)
{
	if (is_training())
	{
		for (size_t i = 0; i < x.memory_size(); i++)
		{
			y.at_index(i) = p(i) * dy.at_index(i);
		}
	}
	return dy;
}

/*************************************************************************************************************************************/
