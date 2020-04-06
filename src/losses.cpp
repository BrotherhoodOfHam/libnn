/*
	Loss functions
*/

#include "nn/losses.h"

using namespace nn;

/*************************************************************************************************************************************/

float mse::forward(const slice& y, const slice& t)
{
	float loss = 0;
	dispatch(y.size(), [&](uint i) {
		// mean squared error
		loss += std::pow(y[i] - t[i], 2)/2;
	});
	return loss;
}

void mse::backward(const slice& y, const slice& t, slice& dy)
{
	dispatch(dy.size(), [&](uint i) {
		// mean squared error
		dy[i] = y[i] - t[i];
	});
}

/*************************************************************************************************************************************/

float binary_cross_entropy::forward(const slice& y, const slice& t)
{
	float loss = 0;
	dispatch(y.size(), [&](uint i) {
		// binary cross-entropy
		loss += -t[i] * std::log(y[i]) - (1.0f - t[i]) * std::log(1.0f - y[i]);
	});
	return loss;
}

void binary_cross_entropy::backward(const slice& y, const slice& t, slice& dy)
{
	dispatch(dy.size(), [&](uint i) {
		// binary cross-entropy
		dy[i] = (y[i] - t[i]) / (y[i] * (1.0f - y[i]) + std::numeric_limits<scalar>::epsilon());
	});
}

/*************************************************************************************************************************************/

float categorical_cross_entropy::forward(const slice& y, const slice& t)
{
	float loss = 0;
	dispatch(y.size(), [&](uint i) {
		// categorical cross-entropy
		loss += -t[i] * std::log(y[i]);
	});
	return loss;
}

void categorical_cross_entropy::backward(const slice& y, const slice& t, slice& dy)
{
	dispatch(dy.size(), [&](uint i) {
		// categorical cross-entropy
		dy[i] = -t[i] / (y[i] + std::numeric_limits<scalar>::epsilon());
	});
}

/*************************************************************************************************************************************/
