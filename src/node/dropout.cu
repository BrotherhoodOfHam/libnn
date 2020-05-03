/*
	Dropout regularization node
*/

#include "device/gpu.h"
#include "nn/node/dropout.h"

using namespace nn;

/*************************************************************************************************************************************/

dropout::dropout(tensor_shape input_shape, float probability) :
	_probability(probability),
	uniform_node(input_shape)
{}

batch dropout::forward(scope& dc, const batch& x)
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
		float p = 1.0f - _probability;
		_dropout = dc.alloc(x.size());
		_rng.random_bernoulli(_dropout, p, 1.0f / p);

		auto y = dc.alloc(x.size());
		dc.vector_mul(y, _dropout, x);
		return y.reshape(x.layout());
	}

	return x;
}

batch dropout::backward(scope& dc, const batch& x, const batch& y, const batch& dy)
{
	if (dc.is_training())
	{
		/*
		dispatch(y.layout(), [&](uint i) {
			dx[i] = p[i] * dy[i];
		});
		*/

		auto dx = dc.alloc(x.size());
		dc.vector_mul(dx, _dropout, dy);
		return dx.reshape(x.layout());
	}

	return dy;
}

/*************************************************************************************************************************************/
