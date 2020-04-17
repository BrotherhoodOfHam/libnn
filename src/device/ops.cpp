/*
	BLAS operations
*/

#include "ops.h"

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

vector nn::vector_mul(context& dc, vector& a, vector& b)
{
	cublas
	cublasDgemm()
}

tensor<2> nn::mat_mul(context& dc, tensor<2>& a, tensor<2>& b)
{

}

/*************************************************************************************************************************************/
