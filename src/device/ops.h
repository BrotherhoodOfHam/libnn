/*
	Tensor operations
*/

#pragma once

#include "kernels.h"
#include "nn/tensors.h"
#include "nn/device.h"

namespace nn
{
	enum class matrix_flag
	{
		transpose_A = 1,
		transpose_B = 2,
		accumulate  = 4
	};
	inline matrix_flag operator|(matrix_flag a, matrix_flag b) { return (matrix_flag)((int)a | (int)b); }
	inline bool operator&(matrix_flag a, matrix_flag b) { return ((int)a & (int)b) != 0; }

	// c[i] = a[i] * b[i]
	void vector_mul(vector& c, const vector& a, const vector& b);

	// c[i] = a[i] + b[i]
	void vector_add(vector& c, const vector& a, const vector& b);

	// c[i] = a[i] - b[i]
	void vector_sub(vector& c, const vector& a, const vector& b);

	scalar vector_sum(const vector& a);

	// c[i,k] = a[i,j] * b[j,k]
	void matrix_mul(tensor<2>& c, const tensor<2>& a, const tensor<2>& b, matrix_flag flags = matrix_flag(0));

	void matrix_set_rows(tensor<2>& m, const vector& a);

	void matrix_sum_rows(vector& sum, const tensor<2>& m);
}
