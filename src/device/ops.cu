/*
	BLAS operations
*/

#include "ops.h"

#include <cublas_v2.h>

using namespace nn;

static cublasHandle_t get_handle()
{
	struct cublas_deleter
	{
		void operator()(cublasHandle_t h)
		{
			if (h != nullptr)
				check(cublasDestroy(h));
		}
	};

	static std::unique_ptr<cublasContext, cublas_deleter> s_handle;
	if (s_handle == nullptr)
	{
		cublasHandle_t h;
		check(cublasCreate(&h));
		s_handle.reset(h);
	}
	return s_handle.get();
}

/*************************************************************************************************************************************/

template<scalar(op)(scalar, scalar)>
__global__ void vector_op_kernel(uint n, scalar* y, const scalar* a, const scalar* b)
{
	uint i = global_index();
	if (i < n)
	{
		y[i] = op(a[i], b[i]);
	}
}

template<scalar(op)(scalar, scalar)>
void launch_vector_op(vector& c, const vector& a, const vector& b)
{
	assert(c.size() == a.size());
	assert(c.size() == b.size());

	int block_size = 256;
	int block_count = (a.size() + block_size - 1) / block_size;

	vector_op_kernel<op><<<block_count, block_size>>>(c.size(), c.ptr(), a.ptr(), b.ptr());
}


__device__ scalar vector_mul_op(scalar a, scalar b) { return a * b; }

void nn::vector_mul(vector& c, const vector& a, const vector& b)
{
	launch_vector_op<vector_mul_op>(c, a, b);
}

__device__ scalar vector_add_op(scalar a, scalar b) { return a * b; }

void nn::vector_add(vector& c, const vector& a, const vector& b)
{
	launch_vector_op<vector_add_op>(c, a, b);
}

__device__ scalar vector_sub_op(scalar a, scalar b) { return a - b; }

void nn::vector_sub(vector& c, const vector& a, const vector& b)
{
	launch_vector_op<vector_sub_op>(c, a, b);
}

/*************************************************************************************************************************************/

scalar nn::vector_sum(const vector& a)
{
	auto r = std::make_unique<scalar>();
	check(cublasSasum(get_handle(), a.size(), a.ptr(), 1, r.get()));
	return *r;
}

/*************************************************************************************************************************************/

/*
// c[i,k] = a[i,j] * b[j,k] + d[i]
__global__ void matrix_mul_add_kernel(uint is, uint js, uint ks, scalar* c, const scalar* a, const scalar* b, const scalar* d)
{
	uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
	uint k = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (i < is && k < ks)
	{
		scalar sum = 0.0f;
		for (uint j = 0; j < js; j++)
		{
			sum += a[js * i + j] * b[ks * j + k];
		}

		c[ks * i + k] = sum + d[i];
	}
}

tensor<2> matrix_muladd(context& dc, const tensor<2>& a, const tensor<2>& b, const vector& d)
{
	uint i = a.shape(0);
	uint j = a.shape(1);
	uint k = b.shape(1);

	assert(b.shape(0) == j);
	assert(d.size() == i);

	auto c = dc.alloc(i, k);

	auto block_size = dim3(256, 256);
	auto block_count = dim3((i + block_size.x - 1) / block_size.x, (k + block_size.y - 1) / block_size.y);

	matrix_mul_add_kernel << <block_count, block_size >> > (i, j, k, c.ptr(), a.ptr(), b.ptr(), d.ptr());

	return c;
}
*/

void nn::matrix_mul(tensor<2>& c, const tensor<2>& a, const tensor<2>& b, matrix_flag flags)
{
	/*
		Because cublas expects column-major matrices when we store them in row-major
		we swap A and B when invoking cublas functions
	*/

	bool trn_A = flags & matrix_flag::transpose_A;
	bool trn_B = flags & matrix_flag::transpose_B;

	auto opA = trn_A ? CUBLAS_OP_T : CUBLAS_OP_N;
	auto opB = trn_B ? CUBLAS_OP_T : CUBLAS_OP_N;

	uint colsA = trn_A ? a.shape(0) : a.shape(1);
	uint rowsA = trn_A ? a.shape(1) : a.shape(0);
	uint colsB = trn_B ? b.shape(0) : b.shape(1);
	uint rowsB = trn_B ? b.shape(1) : b.shape(0);

	// check dimensions
	assert(colsA == rowsB);
	assert(c.shape(0) == rowsA);
	assert(c.shape(1) == colsB);

	uint m = colsB;
	uint n = rowsA;
	uint k = colsA;

	const float alpha = 1.0f;
	const float beta = (flags & matrix_flag::accumulate) ? 1.0f : 0.0f;

	cublasSgemm(
		get_handle(),
		opB,
		opA,
		m, n, k,
		&alpha, b.ptr(), b.shape(1), a.ptr(), a.shape(1),
		&beta, c.ptr(), c.shape(1)
	);
}

/*************************************************************************************************************************************/

__global__ void matrix_set_rows_kernel(uint rows, uint cols, scalar* m, const scalar* v)
{
	uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
	uint j = (blockIdx.y * blockDim.y) + threadIdx.y;
	if ((i < cols) && (j < rows))
	{
		m[(j * cols) + i] = v[i];
	}
}

void nn::matrix_set_rows(tensor<2>& m, const vector& a)
{
	assert(m.shape(1) == a.size());

	dim3 block_size = dim3(32, 32);
	dim3 block_count = dim3(
		(m.shape(1) + block_size.x - 1) / block_size.x, //i - col
		(m.shape(0) + block_size.y - 1) / block_size.y  //j - row
	);
	matrix_set_rows_kernel<<<block_count, block_size>>>(m.shape(0), m.shape(1), m.ptr(), a.ptr());
}

/*************************************************************************************************************************************/

__global__ void matrix_sum_rows_kernel(uint rows, uint cols, const scalar* m, scalar* v)
{
	uint i = (blockIdx.x * blockDim.x) + threadIdx.x;

	scalar sum = 0;
	for (uint j = 0; j < rows; j++)
	{
		sum += m[(j * cols) + i];
	}

	v[i] = sum;
}

void nn::matrix_sum_rows(vector& sum, const tensor<2>& m)
{
	assert(m.shape(1) == sum.size());

	uint block_size = 64;
	uint block_count = (m.shape(1) + block_size - 1) / block_size;
	matrix_sum_rows_kernel<<<block_count, block_size>>>(m.shape(0), m.shape(1), m.ptr(), sum.ptr());
}

/*************************************************************************************************************************************/
