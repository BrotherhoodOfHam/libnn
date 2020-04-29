/*
	Implementation of Stochastic gradient descent algorithm 
*/

#include "device/kernels.h"
#include "nn/optimizers.h"

using namespace nn;

/*************************************************************************************************************************************/

__global__ void sgd_kernel(uint n, scalar* p, scalar* grad, float k)
{
	uint i = global_index();
	if (i < n)
	{
		p[i] -= k * grad[i];
	}
}

__global__ void sgd_moment_kernel(uint n, scalar* p, scalar* v1, scalar* v0, float m, float k)
{
	uint i = global_index();
	if (i < n)
	{
		v0[i] = (m * v0[i]) + (1 - m) * v1[i];
		p[i] -= k * v0[i];
	}
}

/*************************************************************************************************************************************/

struct sgd::function : public opt_function::state
{
	float _k;

	function(uint param_size, float k) : _k(k) {}

	void call(vector& param, const vector& grad)
	{
		/*
		dispatch(parameter.size(), [&](uint i) {
			p[i] -= _k * grad[i];
		});
		*/

		int block_size = 256;
		int block_count = (param.size() + block_size - 1) / block_size;
		sgd_kernel<<<block_count, block_size>>>(param.size(), param.ptr(), grad.ptr(), _k);
	}
};

struct sgd::function_with_momentum : public opt_function::state
{
	float _k;
	float _m;
	buffer _gradient;

	function_with_momentum(uint param_size, float k, float m) :
		_k(k), _m(m), _gradient(param_size)
	{
		//tensor_zero(_gradient.as_vector());
		auto& dc = context::get_global();
		dc.zero(_gradient.as_vector());
	}

	void call(vector& parameter, const vector& gradient)
	{
		assert(parameter.size() == _gradient.size());

		auto p = parameter;
		auto v1 = gradient;
		auto v0 = _gradient.as_vector(); //previous gradient

		/*
		dispatch(parameter.size(), [&](uint i) {
			v0[i] = (_m * v0[i]) + (1 - _m) * v1[i];
			p[i] -= _k * v0[i];
		});
		*/

		int block_size = 256;
		int block_count = (p.size() + block_size - 1) / block_size;
		sgd_moment_kernel<<<block_count, block_size>>>(parameter.size(), parameter.ptr(), v1.ptr(), v0.ptr(), _m, _k);
	}
};


/*************************************************************************************************************************************/

opt_function sgd::for_param(uint param_size) const
{
	if (_m == 0.0f)
	{
		return opt_function::make<sgd::function>(param_size, _k);
	}
	return opt_function::make<sgd::function_with_momentum>(param_size, _k, _m);
}

/*************************************************************************************************************************************/
