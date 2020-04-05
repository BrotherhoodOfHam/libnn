/*
	Implementation of Stochastic gradient descent algorithm 
*/

#include "nn/optimizers.h"

using namespace nn;

/*************************************************************************************************************************************/

struct function : public opt_function::state
{
	float _k;

	function(uint param_size, float k) : _k(k) {}

	void call(buffer & parameter, const buffer & gradient)
	{
		auto p = parameter.as_vector();
		auto grad = gradient.as_vector();

		dispatch(parameter.size(), [&](uint i) {
			p[i] -= _k * grad[i];
		});
	}
};

struct function_with_momentum : public opt_function::state
{
	float _k;
	float _m;
	buffer _gradient;

	function_with_momentum(uint param_size, float k, float m) :
		_k(k), _m(m), _gradient(param_size)
	{
		tensor_zero(_gradient.as_vector());
	}

	void call(buffer& parameter, const buffer& gradient)
	{
		assert(parameter.size() == _gradient.size());

		auto p = parameter.as_vector();
		auto v1 = gradient.as_vector();
		auto v0 = _gradient.as_vector(); //previous gradient

		dispatch(parameter.size(), [&](uint i) {
			v0[i] = (_m * v0[i]) + (1 - _m) * v1[i];
			p[i] -= _k * v0[i];
		});
	}
};


/*************************************************************************************************************************************/

opt_function sgd::for_param(uint param_size) const
{
	if (_m == 0.0f)
	{
		return opt_function::make<function>(param_size, _k);
	}
	return opt_function::make<function_with_momentum>(param_size, _k, _m);
}

/*************************************************************************************************************************************/
