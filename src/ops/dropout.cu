/*
	Dropout regularization node
*/

#include "device/kernels.h"
#include "device/ops.h"
#include "nn/ops/dropout.h"

using namespace nn;

/*************************************************************************************************************************************/

dropout::dropout(tensor_shape input_shape, float probability) :
	_probability(probability),
	uniform_node(input_shape)
{}

vector dropout::forward(context& dc, const vector& x)
{
	if (dc.is_training())
	{
		/*
		auto rng = new_random_engine();
		dispatch(y.layout(), [&](uint i) {
			std::bernoulli_distribution dist(_probability);
			p[i] = dist(rng) ? 0.0f : 1.0f;
			y[i] = p[i] * x[i];
		});
		*/

		// generate random dropout values
		_dropout = dc.alloc(x.size());
		_rng.random_bernoulli(_dropout, 1.0f - _probability);

		auto y = dc.alloc(x.size());
		vector_mul(y, _dropout, x);
		return y;
	}

	return x;
}

vector dropout::backward(context& dc, const vector& x, const vector& dy)
{
	if (dc.is_training())
	{
		/*
		dispatch(y.layout(), [&](uint i) {
			dx[i] = p[i] * dy[i];
		});
		*/

		auto dx = dc.alloc(x.size());
		vector_mul(dx, _dropout, dy);
		return dx;
	}

	return dy;
}

/*************************************************************************************************************************************/
