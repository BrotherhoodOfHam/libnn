/*
	Implementation of Adam optimizer
*/

#include "nn/optimizers.h"

using namespace nn;

/*************************************************************************************************************************************/

struct function : public opt_function::state
{
	float _alpha;
	float _beta1;
	float _beta2;
	float _epsilon;

	uint      _timestep = 0;
	buffer    _moments;
	layout<2> _layout;

public:

	function(uint parameters_size, float alpha = 0.01f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8) :
		_timestep(0), _alpha(alpha), _beta1(beta1), _beta2(beta2), _epsilon(epsilon), _layout(parameters_size, 2), _moments(parameters_size * 2)
	{
		tensor_zero(_moments.as_vector());
	}

	void call(buffer& parameter, const buffer& gradient)
	{
		auto p = parameter.as_vector();
		auto grad = gradient.as_vector();
		auto mom = _moments.as_tensor(_layout);
		_timestep++;

		// earlier timesteps are biased because the moments are close to 0
		scalar m_t_scale = 1.0f / (scalar)(1 - std::pow(_beta1, _timestep));
		scalar v_t_scale = 1.0f / (scalar)(1 - std::pow(_beta2, _timestep));

		dispatch(parameter.size(), [&](uint i) {
			scalar g = grad[i];
			scalar m_t = (_beta1 * mom[i][0]) + (1 - _beta1) * g;		//moving average of gradient
			scalar v_t = (_beta2 * mom[i][1]) + (1 - _beta2) * (g * g); //moving average of gradient^2
			// correct bias
			m_t = m_t * m_t_scale;
			v_t = v_t * v_t_scale;
			// update parameter
			scalar v = m_t / (std::sqrt(v_t) + _epsilon);
			p[i] -= _alpha * v;
			mom[i][0] = m_t;
			mom[i][1] = v_t;
		});
	}
};
/*************************************************************************************************************************************/

opt_function adam::for_param(uint param_size) const
{
	return opt_function::make<function>(param_size, _alpha, _beta1, _beta2, _epsilon);
}

/*************************************************************************************************************************************/
