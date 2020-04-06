/*
	Loss functions
*/

#pragma once

#include "tensors.h"

namespace nn
{
	class loss_function
	{
		using fptr = float(*)(const slice & y, const slice & t);
		using dfptr = void(*)(const slice & y, const slice & t, slice & dy);

		fptr _func;
		dfptr _deriv;

	public:

		loss_function(fptr func, dfptr deriv) :
			_func(func), _deriv(deriv)
		{}

		float operator()(const slice& y, const slice& t) const
		{
			return _func(y, t);
		}

		void grad(const slice& y, const slice& t, slice& dy) const
		{
			_deriv(y, t, dy);
		}
	};

	template<class F>
	class basic_loss_function : public loss_function
	{
	public:

		basic_loss_function() : loss_function(&F::forward, &F::backward) {}
	};

	class mse : public basic_loss_function<mse>
	{
	public:

		static float forward(const slice& y, const slice& t);
		static void backward(const slice& y, const slice& t, slice& dy);
	};

	class binary_cross_entropy : public basic_loss_function<binary_cross_entropy>
	{
	public:

		static float forward(const slice& y, const slice& t);
		static void backward(const slice& y, const slice& t, slice& dy);
	};

	class categorical_cross_entropy : public basic_loss_function<categorical_cross_entropy>
	{
	public:

		static float forward(const slice& y, const slice& t);
		static void backward(const slice& y, const slice& t, slice& dy);
	};
}
