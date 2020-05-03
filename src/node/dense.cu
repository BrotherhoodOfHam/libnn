/*
	Simple fully connected layer
*/

#include <random>

#include "device/gpu.h"
#include "nn/node/dense.h"

using namespace nn;

/*************************************************************************************************************************************/

dense_layer::dense_layer(tensor_shape input_shape, uint layer_size) :
	_input(input_shape), _output(layer_size),
	_w(layer_size, input_shape[0]),
	_b(layer_size)
{
	const float sqrtn = std::sqrt((float)input_shape[0]);

	auto& d = device::get();
	d.zero(_b.v());
	d.zero(_b.dv());
	d.zero(_w.dv());
	d.random_normal(_w.v(), 1.0f / sqrtn);
}

batch dense_layer::forward(scope& dc, const batch& x)
{
	auto y = dc.batch_alloc(_output);

	dc.matrix_set_rows(y, _b.v());
	dc.matrix_mul(y, x, _w.v(), ops::accumulate | ops::transpose_B);

	return y;

	/*
	//for each row:
	//y = w.x + b
	dispatch(y.layout(), [&](uint n, uint j) {
		//z = w.x + b
		scalar z = 0.0f;
		for (uint i = 0; i < w.shape(1); i++)
			z += x[n][i] * w[j][i];
		z += b[j];
		y[n][j] = z;
	});

	return y.data();
	*/
}

batch dense_layer::backward(scope& dc, const batch& x, const batch& y, const batch& dy)
{
	auto dx = dc.batch_alloc(_input);

	tensor<2> dw = _w.dv();
	vector    db = _b.dv();

	// compute partial derivatives w.r.t weights
	// δw = δy^T * x  (outer product)
	dc.matrix_mul(dw, dy, x, ops::transpose_A);

	// compute partial derivatives w.r.t biases
	// δb = δy
	dc.matrix_sum_rows(db, dy);

	// δx = δy * w^T 
	dc.matrix_mul(dx, dy, _w.v());

	return dx;

	/*
	// δ/δy = partial derivative of loss w.r.t to output
	// the goal is to find the derivatives w.r.t to parameters: δw, δb, δx
	const uint batch_size = dc.batch_size();

	// compute partial derivatives w.r.t biases
	dispatch(b.layout(), [&](uint i) {
		// δb = δy
		scalar sum = 0;
		for (uint b = 0; b < batch_size; b++)
			sum += dy[b][i];
		db[i] = sum;
	});

	// compute partial derivatives w.r.t weights
	dispatch(w.layout(), [&](uint j, uint i) {
		// δw = δy * x (outer product)
		scalar sum = 0;
		for (uint b = 0; b < batch_size; b++)
			sum += dy[b][j] * x[b][i];
		dw[j][i] = sum;
	});

	// compute partial derivative w.r.t input x
	dispatch(dx.layout(), [&](uint b, uint i) {
		// δx = w^T * δy
		scalar sum = 0;
		for (uint j = 0; j < w.shape(0); j++)
			sum += w[j][i] * dy[b][j];
		dx[b][i] = sum;
	});

	return dx.data();
	*/
}

/*************************************************************************************************************************************/
