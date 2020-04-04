/*
	Optimization functions
*/

#pragma once

#include <functional>

#include "tensors.h"

namespace nn
{
	/*
		Generic optimization function
	*/
	using optimization_function = std::function<void(buffer& param, const buffer& gradient)>;

	/*
		Stochastic gradient descent optimizer
	*/
	class sgd
	{
		float _k;
		float _b;
		buffer _gradient;

	public:

		sgd(uint parameters_size, float learning_rate, float momentum = 0.0f) :
			_k(learning_rate), _b(momentum), _gradient(parameters_size)
		{
			tensor_zero(_gradient.as_vector());
		}

		void operator()(buffer& parameter, const buffer& gradient)
		{
			auto p = parameter.as_vector();
			auto v1 = gradient.as_vector();
			auto v0 = _gradient.as_vector(); //previous gradient

			dispatch(parameter.size(), [&](uint i) {
				float v = (_b * v0[i]) + (1 - _b) * v1[i];
				p[i] -= _k * v;
				v0[i] = v;
			});
		}
	};

}
