/*
	Loss functions
*/

#include "device/gpu.h"
#include "nn/losses.h"

using namespace nn;

/*************************************************************************************************************************************/

using kernel_ptr = scalar(*)(scalar y, scalar t);

template<kernel_ptr loss_func>
__global__ void loss_kernel(uint n, const scalar* y, const scalar* t, scalar* dy)
{
	uint i = global_index();
	if (i < n)
	{
		dy[i] = loss_func(y[i], t[i]);
	}
}

template<kernel_ptr loss_func>
static void launch_loss(const vector& y, const vector& t, vector& dy)
{
	assert(y.size() == t.size());
	assert(y.size() == dy.size());

	uint block_size = 256;
	uint block_count = (dy.size() + block_size - 1) / block_size;
	loss_kernel<loss_func><<<block_count, block_size>>>(dy.size(), y.ptr(), t.ptr(), dy.ptr());
}

/*************************************************************************************************************************************/

__device__ scalar mse_loss(scalar y, scalar t)
{
	scalar d = y - t;
	return (d * d) / 2;
}

float mse::forward(scope& dc, const vector& y, const vector& t)
{
	/*
	dispatch(y.size(), [&](uint i) {
		// mean squared error
		loss += std::pow(y[i] - t[i], 2)/2;
	});
	*/

	auto losses = dc.alloc(y.size());
	launch_loss<mse_loss>(y, t, losses);
	return dc.vector_sum(losses);
}

vector mse::backward(scope& dc, const vector& y, const vector& t)
{
	/*
	dispatch(dy.size(), [&](uint i) {
		// mean squared error
		dy[i] = y[i] - t[i];
	});
	*/

	auto dy = dc.alloc(y.size());
	dc.vector_sub(dy, y, t);
	return dy;
}

/*************************************************************************************************************************************/

__device__ scalar bce_kernel(scalar y, scalar t)
{
	return -t * std::log(y) - (1.0f - t) * std::log(1.0f - y);
}

__device__ scalar bce_grad_kernel(scalar y, scalar t)
{
	return (y - t) / (y * (1.0f - y) + FLT_EPSILON);
}

float binary_cross_entropy::forward(scope& dc, const vector& y, const vector& t)
{
	/*
	dispatch(y.size(), [&](uint i) {
		// binary cross-entropy
		loss += -t[i] * std::log(y[i]) - (1.0f - t[i]) * std::log(1.0f - y[i]);
	});
	*/

	auto losses = dc.alloc(y.size());
	launch_loss<bce_kernel>(y, t, losses);
	return dc.vector_sum(losses);
}

vector binary_cross_entropy::backward(scope& dc, const vector& y, const vector& t)
{
	/*
	dispatch(dy.size(), [&](uint i) {
		// binary cross-entropy
		dy[i] = (y[i] - t[i]) / (y[i] * (1.0f - y[i]) + std::numeric_limits<scalar>::epsilon());
	});
	*/

	auto dy = dc.alloc(y.size());
	launch_loss<bce_grad_kernel>(y, t, dy);
	return dy;
}

/*************************************************************************************************************************************/

__device__ scalar cce_kernel(scalar y, scalar t)
{
	return -t * std::log(y + FLT_EPSILON);
}

__device__ scalar cce_grad_kernel(scalar y, scalar t)
{
	return -t / (y + FLT_EPSILON);
}

float categorical_cross_entropy::forward(scope& dc, const vector& y, const vector& t)
{
	/*
	dispatch(y.size(), [&](uint i) {
		// categorical cross-entropy
		loss += -t[i] * std::log(y[i]);
	});
	*/

	auto losses = dc.alloc(y.size());
	launch_loss<cce_kernel>(y, t, losses);
	return dc.vector_sum(losses);
}

vector categorical_cross_entropy::backward(scope& dc, const vector& y, const vector& t)
{
	/*
	dispatch(dy.size(), [&](uint i) {
		// categorical cross-entropy
		dy[i] = -t[i] / (y[i] + std::numeric_limits<scalar>::epsilon());
	});
	*/
	auto dy = dc.alloc(y.size());
	launch_loss<cce_grad_kernel>(y, t, dy);
	return dy;
}

/*************************************************************************************************************************************/
