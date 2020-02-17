/*
	Dropout regularization node
*/

#include "dropout.h"

using namespace nn;
using namespace nn::nodes;

/*************************************************************************************************************************************/

dropout::dropout(size_t input_size, float probability) :
	_probability(probability),
	node(input_size, input_size)
{
}

const tensor& dropout::forward(const tensor& x)
{
	return x;
}

const tensor& dropout::backward(const tensor& x, const tensor& dy)
{
	return dy;
}

/*************************************************************************************************************************************/
