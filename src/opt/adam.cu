/*
	Implementation of Adam optimizer
*/

#include "device/kernels.h"
#include "nn/optimizers.h"

using namespace nn;

/*************************************************************************************************************************************/

__global__ void adam_kernel(
	uint n, scalar* p, scalar* grad, scalar* moments,
	float alpha, float beta1, float beta2, float epsilon,
	float beta1_t, float beta2_t
)
{
	uint i = global_index();
	if (i < n)
	{
		scalar g = grad[i];
		scalar m = moments[2*i+0] = (beta1 * moments[2*i+0]) + ((1 - beta1) * g);	   // update moving average of gradient
		scalar v = moments[2*i+1] = (beta2 * moments[2*i+1]) + ((1 - beta2) * g * g); // update moving average of gradient^2
		// correct bias
		scalar m_t = m / (1 - beta1_t);
		scalar v_t = v / (1 - beta2_t);
		// update parameter
		p[i] -= alpha * (m_t / (std::sqrt(v_t + epsilon)));
	}
}

/*************************************************************************************************************************************/

struct adam::function : public opt_function::state
{
	float _alpha;
	float _beta1;
	float _beta2;
	float _epsilon;

	float     _beta1_t; //beta1 at time t
	float	  _beta2_t; //beta2 at time t
	buffer    _moments;
	tensor_layout<2> _layout;

public:

	function(uint parameters_size, float alpha = 0.01f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8) :
		_alpha(alpha), _beta1(beta1), _beta2(beta2), _beta1_t(beta1), _beta2_t(beta2), _epsilon(epsilon),
		_layout(parameters_size, 2), _moments(parameters_size * 2)
	{
		//tensor_zero(_moments.as_vector());
		auto& dc = context::get_global();
		dc.zero(_moments.as_vector());
	}

	void call(buffer& parameter, const buffer& gradient)
	{
		auto param = parameter.as_vector();
		auto grad = gradient.as_vector();
		auto mts = _moments.as_tensor(_layout);

		/*
		dispatch(parameter.size(), [&](uint i) {
			scalar g = grad[i];
			mts[i][0] = (_beta1 * mts[i][0]) + ((1 - _beta1) * g);	   // update moving average of gradient
			mts[i][1] = (_beta2 * mts[i][1]) + ((1 - _beta2) * g * g); // update moving average of gradient^2
			// correct bias
			scalar m_t = mts[i][0] / (1 - _beta1_t);
			scalar v_t = mts[i][1] / (1 - _beta2_t);
			// update parameter
			param[i] -= _alpha * (m_t / (std::sqrt(v_t + _epsilon)));
		});
		*/

		int block_size = 256;
		int block_count = (param.size() + block_size - 1) / block_size;
		adam_kernel<<<block_count, block_size>>>(param.size(), param.ptr(), grad.ptr(), mts.ptr(), _alpha, _beta1, _beta2, _epsilon, _beta1_t, _beta2_t);

		_beta1_t *= _beta1;
		_beta2_t *= _beta2;
	}
};

/*************************************************************************************************************************************/

opt_function adam::for_param(uint param_size) const
{
	return opt_function::make<adam::function>(param_size, _alpha, _beta1, _beta2, _epsilon);
}

/*************************************************************************************************************************************/
