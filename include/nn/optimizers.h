/*
	Optimization functions
*/

#pragma once

#include <functional>

#include "tensors.h"
#include "ops/node.h"

namespace nn
{
	/* 
		Optimization function interface
	*/
	class opt_function
	{
	public:

		struct state
		{
			virtual ~state() {}
		};
		using fptr = void (*)(state* state, buffer& param, const buffer& gradient);

	private:

		std::unique_ptr<state> _state;
		fptr                   _func;

		opt_function(fptr func, std::unique_ptr<state>&& st = std::unique_ptr<state>()) :
			_state(std::move(st)), _func(func)
		{}

	public:

		opt_function() : _func(nullptr) {}
		opt_function(opt_function&&) = default;

		void operator()(buffer& param, const buffer& gradient)
		{
			return _func(_state.get(), param, gradient);
		}

		// helper function for creating optimizer functions
		template<class state_impl, class = std::enable_if_t<std::is_convertible_v<state_impl*, state*>>, class ... args_type>
		static opt_function make(args_type&& ... args)
		{
			fptr thunk = [](state* s, buffer& param, const buffer& grad)
			{
				((state_impl*)s)->call(param, grad);
			};
			return opt_function(thunk, std::make_unique<state_impl>(args...));
		}
	};

	/*
		Optimizer type interface
		Can be used to construct optimizers for parameters
	*/
	class optimizer_type
	{
	public:

		virtual opt_function for_param(uint param_size) const = 0;
	};

	/*
		Stochastic gradient descent optimizer
	*/
	class sgd : public optimizer_type
	{
		struct function;
		struct function_with_momentum;

		float _k;
		float _m;

	public:

		sgd(float learning_rate=0.01f, float momentum = 0.0f) : _k(learning_rate), _m(momentum) {}

		opt_function for_param(uint param_size) const override;
	};

	/*
		Adam optimizer
	*/
	class adam : public optimizer_type
	{
		struct function;

		float _alpha;
		float _beta1;
		float _beta2;
		float _epsilon;

	public:

		adam(float alpha = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8) :
			_alpha(alpha), _beta1(beta1), _beta2(beta2), _epsilon(epsilon)
		{}

		opt_function for_param(uint param_size) const override;
	};

}
