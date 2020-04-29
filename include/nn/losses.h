/*
	Loss functions
*/

#pragma once

#include "device.h"

namespace nn
{
	class loss_function
	{
		using fptr = float(*)(scope& dc, const vector &y, const vector &t);
		using dfptr = vector(*)(scope& dc, const vector &y, const vector &t);

		fptr _func;
		dfptr _deriv;

	public:

		loss_function(fptr func, dfptr deriv) :
			_func(func), _deriv(deriv)
		{}

		float loss(scope& dc, const vector& y, const vector& t) const
		{
			return _func(dc, y, t);
		}

		vector grad(scope& dc, const vector& y, const vector& t) const
		{
			return _deriv(dc, y, t);
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

		static float forward(scope& dc, const vector& y, const vector& t);
		static vector backward(scope& dc, const vector& y, const vector& t);
	};

	class binary_cross_entropy : public basic_loss_function<binary_cross_entropy>
	{
	public:

		static float forward(scope& dc, const vector& y, const vector& t);
		static vector backward(scope& dc, const vector& y, const vector& t);
	};

	class categorical_cross_entropy : public basic_loss_function<categorical_cross_entropy>
	{
	public:

		static float forward(scope& dc, const vector& y, const vector& t);
		static vector backward(scope& dc, const vector& y, const vector& t);
	};
}
