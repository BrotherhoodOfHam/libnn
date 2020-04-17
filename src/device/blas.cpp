/*
	BLAS operations
*/

#include "blas.h"

#include <cublas_v2.h>

using namespace nn;

static cublasHandle_t get_handle()
{
	static cublasHandle_t s_handle = nullptr;
	if (s_handle == nullptr)
	{
		check(cublasCreate(&s_handle));
	}
}

/*************************************************************************************************************************************/

vector blas::gemv(context& dc, vector& a, vector& b)
{
	cublasSetMatrix()
}

tensor<2> blas::gemm(context& dc, tensor<2>& a, tensor<2>& b)
{

}

/*************************************************************************************************************************************/
