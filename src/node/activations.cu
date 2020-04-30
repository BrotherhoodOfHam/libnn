/*
	Activation functions
*/

#include <cmath>
#include <algorithm>

#include "device/gpu.h"
#include "nn/node/activations.h"

using namespace nn;

/*************************************************************************************************************************************/

template<scalar(activate)(scalar)>
__global__ void activation_kernel(uint n, scalar* y, const scalar* x)
{
	uint i = global_index();
	if (i < n)
	{
		y[i] = activate(x[i]);
	}
}

template<scalar(activate_d)(scalar, scalar, scalar)>
__global__ void activation_d_kernel(uint n, scalar* dx, const scalar* x, const scalar* y, const scalar* dy)
{
	uint i = global_index();
	if (i < n)
	{
		dx[i] = activate_d(x[i], y[i], dy[i]);
	}
}

template<scalar(func)(scalar)>
static vector launch(scope& dc, const vector& x)
{
	auto y = dc.alloc(x.size());

	uint block_size = 256;
	uint block_count = (x.size() + block_size - 1) / block_size;
	activation_kernel<func><<<block_count, block_size>>>(x.size(), y.ptr(), x.ptr());

	return y;
}

template<scalar(func)(scalar, scalar, scalar)>
static vector launch(scope& dc, const vector& x, const vector& y, const vector& dy)
{
	auto dx = dc.alloc(x.size());

	uint block_size = 256;
	uint block_count = (x.size() + block_size - 1) / block_size;

	activation_d_kernel<func><<<block_count, block_size>>>(x.size(), dx.ptr(), x.ptr(), y.ptr(), dy.ptr());

	return dx;
}

/*************************************************************************************************************************************/

__device__ scalar sigmoid_func(scalar x)
{
	return 1.0f / (1.0f + std::exp(-x));
}

__device__ scalar sigmoid_d_func(scalar x, scalar y, scalar dy)
{
	return (y * (1.0f - y)) * dy;
}

vector activation::sigmoid::forward(scope& dc, const vector& x)
{
	/*
	return activate(x, [](scalar x) {
		return 1.0f / (1.0f + std::exp(-x));
	});
	*/
	return launch<sigmoid_func>(dc, x);
}

vector activation::sigmoid::backward(scope& dc, const vector& x, const vector& y, const vector& dy)
{
	/*
	// σ'(x) = σ(x) * (1 - σ(x))
	return derivative(x, dy, [](scalar x, scalar y, scalar dy, scalar dx) {
		return (y * (1.0f - y)) * dy;
	});
	*/
	return launch<sigmoid_d_func>(dc, x, y, dy);
}

/*************************************************************************************************************************************/

__device__ scalar tanh_func(scalar x)
{
	scalar a = std::exp(x);
	scalar b = 1.0f / a;
	return (a - b) / (a + b);
}

__device__ scalar tanh_d_func(scalar x, scalar y, scalar dy)
{
	return (1 - (y * y)) * dy;
}

vector activation::tanh::forward(scope& dc, const vector& x)
{
	/*
	return activate(x, [](scalar x) {
		scalar a = std::exp(x);
		scalar b = 1.0f / a;
		return (a - b) / (a + b);
	});
	*/
	return launch<tanh_func>(dc, x);
}

vector activation::tanh::backward(scope& dc, const vector& x, const vector& y, const vector& dy)
{
	/*
	// tanh'(x) = 1 - tanh(x)^2
	return derivative(x, dy, [](scalar x, scalar y, scalar dy, scalar dx) {
		return (1 - (y * y)) * dy;
	});
	*/
	return launch<tanh_d_func>(dc, x, y, dy);
}

/*************************************************************************************************************************************/

__device__ scalar relu_func(scalar x)
{
	return (x > 0.0f) ? x : 0.0f;
}

__device__ scalar relu_d_func(scalar x, scalar y, scalar dy)
{
	return (x > 0.0f) ? dy : 0;
}

vector activation::relu::forward(scope& dc, const vector& x)
{
	/*
	return activate(x, [](scalar x) {
		return std::max(x, 0.0f);
	});
	*/
	return launch<relu_func>(dc, x);
}

vector activation::relu::backward(scope& dc, const vector& x, const vector& y, const vector& dy)
{
	/*
	return derivative(x, dy, [](scalar x, scalar y, scalar dy, scalar dx) {
		return (x > 0.0f) ? dy : 0;
	});
	*/
	return launch<relu_d_func>(dc, x, x, dy);
}

/*************************************************************************************************************************************/

__global__ void leaky_relu_kernel(uint n, scalar* y, const scalar* x, scalar leakiness)
{
	uint i = global_index();
	if (i < n)
	{
		y[i] = (x[i] > 0) ? x[i] : leakiness * x[i];
	}
}

__global__ void leaky_relu_d_kernel(uint n, scalar* dx, const scalar* x, const scalar* dy, scalar leakiness)
{
	uint i = global_index();
	if (i < n)
	{
		dx[i] = ((x[i] > 0) ? 1.0f : leakiness) * dy[i];
	}
}

vector activation::leaky_relu::forward(scope& dc, const vector& x)
{
	/*
	return activate(x, [=](scalar x) {
		return (x > 0) ? x : _leakiness * x;
	});
	*/
	auto y = dc.alloc(x.size());

	uint block_size = 256;
	uint block_count = (x.size() + block_size - 1) / block_size;

	leaky_relu_kernel<<<block_count, block_size>>>(x.size(), y.ptr(), x.ptr(), _leakiness);

	return y;
}

vector activation::leaky_relu::backward(scope& dc, const vector& x, const vector& y, const vector& dy)
{
	/*
	return derivative(x, dy, [=](scalar x, scalar y, scalar dy, scalar dx) {
		return ((x > 0) ? 1.0f : _leakiness) * dy;
	});
	*/
	auto dx = dc.alloc(x.size());

	uint block_size = 256;
	uint block_count = (x.size() + block_size - 1) / block_size;

	leaky_relu_d_kernel<<<block_count, block_size>>>(x.size(), dx.ptr(), x.ptr(), dy.ptr(), _leakiness);

	return dx;
}

/*************************************************************************************************************************************/

__global__ void softmax_kernel(uint b, uint n, scalar* _y, const scalar* _x)
{
	uint i_b = global_index();

	if (i_b < b)
	{
		scalar* y = _y + (i_b * n);
		const scalar* x = _x + (i_b * n);

		scalar max = x[0];
		for (uint i = 0; i < n; i++)
			if (x[i] > max)
				max = x[i];

		scalar sum = 0.0f;
		for (uint i = 0; i < n; i++)
		{
			scalar a = std::exp(x[i] - max);
			sum += a;
			y[i] = a;
		}

		for (uint i = 0; i < n; i++)
		{
			y[i] /= sum;
		}
	}
}

__global__ void softmax_d_kernel(uint b, uint n, scalar* _dx, const scalar* _y, const scalar* _dy)
{
	uint i_b = global_index();

	if (i_b < b)
	{
		scalar* dx = _dx + (i_b * n);
		const scalar* dy = _dy + (i_b * n);
		const scalar* y = _y + (i_b * n);

		for (uint j = 0; j < n; j++)
		{
			float sum = 0.0f;
			for (uint i = 0; i < n; i++)
			{
				float dz = (i == j)
					? y[i] * (1 - y[i])
					: -y[i] * y[j];

				sum += dy[i] * dz;
			}
			dx[j] = sum;
		}
	}
}

vector activation::softmax::forward(scope& dc, const vector& x)
{
	/*
	auto x = _x.as_vector();

	scalar max = x[0];
	for (uint i = 0; i < x.size(); i++)
		if (x[i] > max)
			max = x[i];

	scalar sum = 0.0f;
	for (uint i = 0; i < x.size(); i++)
	{
		scalar a = std::exp(x[i] - max);
		sum += a;
		y[i] = a;
	}

	for (uint i = 0; i < x.size(); i++)
	{
		y[i] /= sum;
	}

	return y.data();
	*/

	auto y = dc.alloc(x.size());

	uint block_size = 32;
	uint block_count = (x.size() + block_size - 1) / block_size;
	
	uint b = dc.batch_size();
	uint n = x.size() / b;
	softmax_kernel<<<block_count, block_size>>>(b, n, y.ptr(), x.ptr());

	return y;
}

vector activation::softmax::backward(scope& dc, const vector& x, const vector& y, const vector& dy)
{
	/*
	auto dy = _dy.as_vector();
	for (uint j = 0; j < x.size(); j++)
	{
		float sum = 0.0f;
		for (uint i = 0; i < x.size(); i++)
		{
			float dz = (i == j)
				? y[i] * (1 - y[i])
				: -y[i] * y[j];

			sum += dy[i] * dz;
		}
		dx[j] = sum;
	}

	return dx.data();
	*/

	auto dx = dc.alloc(x.size());

	uint b = dc.batch_size();
	uint n = x.size() / b;

	uint block_size = 32;
	uint block_count = (b + block_size - 1) / block_size;

	softmax_d_kernel<<<block_count, block_size>>>(b, n, dx.ptr(), y.ptr(), dy.ptr());

	return dx;
}

/*************************************************************************************************************************************/
