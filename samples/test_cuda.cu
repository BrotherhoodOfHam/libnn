#include <iostream>

#include <algorithm>
#include <string>
#include <array>
#include <vector>
#include <memory>

#include <cuda.h>
#include <curand_kernel.h>
#include <cublas_v2.h>

using namespace std;

template<class error_type>
void check(error_type error)
{
    if (error)
    {
        std::string msg = "cudaError: "s + std::to_string(error);
        std::cerr << msg << std::endl;
        throw std::runtime_error(msg);
    }
}

#define ASSERT(cond) if (!(cond)) throw std::runtime_error(#cond)

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ int global_index()
{
    return (blockIdx.x * blockDim.x) + threadIdx.x;
}

// function to add the elements of two arrays
__global__
void add(int n, float* x, float* y)
{
    int stride = gridDim.x * blockDim.x;
    int index = (blockDim.x * blockIdx.x) + threadIdx.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

__global__
void monolithic_add(int n, float* x, float* y)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < n)
        y[i] = x[i] + y[i];
}

int main2(void)
{
    int N = 1 << 20; // 1M elements

    float* x = nullptr;
    float* y = nullptr;
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }


    int blockSize = 256;
    int blockCount = (N + blockSize - 1) / blockSize;

    // Run kernel on 1M elements on the CPU
    add<<<blockCount, blockSize>>> (N, x, y);
    //monolithic_add<<<>>>

    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

using scalar = float;
using uint = unsigned;
using byte = unsigned char;

class slice
{
	scalar* _ptr;
	uint    _size;

public:

    __host__ __device__ inline slice(float* ptr, const uint* shape, const uint* strides) :
        _ptr(ptr), _size(shape[0])
    {}

    __host__ __device__
    slice(scalar* ptr, uint size) :
		_ptr(ptr), _size(size)
	{}

    //__host__ __device__ slice(slice&&) = default;
    //__host__ __device__ slice(const slice&) = default;

    __host__ __device__ scalar* ptr() const { return _ptr; }
    __host__ __device__ uint size() const { return _size; }

    __host__ __device__ scalar& at(uint index) const
	{
		return _ptr[index];
	}

    __host__ __device__ scalar& operator[](uint i) const { return at(i); }
};

template<uint dims>
class tensor_slice;

template<>
class tensor_slice<1> : public slice
{
public:
	using slice::slice;
};

template<uint dims>
class tensor_slice
{
	static_assert(dims > 0, "tensor rank must be greater than 0");

	scalar*     _ptr;
    const uint* _shape;
    const uint* _strides;

public:

    __host__ __device__ inline tensor_slice(scalar* ptr, const uint* shape, const uint* strides) :
        _ptr(ptr), _shape(shape), _strides(strides)
    {}

    //__host__ __device__
    tensor_slice(const tensor_slice&) = default;

    __host__ __device__ inline scalar* ptr() const { return _ptr; }
    __host__ __device__ inline uint size() const { return _shape[0]; }
    __host__ __device__ inline uint total_size() const { return _shape[0] * _strides[0]; }
    __host__ __device__ inline constexpr uint shape(uint i) const { return _shape[i]; }

    __host__ __device__ inline tensor_slice<dims - 1> operator[](uint index) const
	{
		return tensor_slice<dims - 1>(
			_ptr + (index * _strides[0]),
			_shape + 1,
			_strides + 1
		);
	}
};

template<uint dims, class = std::enable_if_t<(dims > 0)>>
class layout
{
    std::array<uint, dims> _shape;
    std::array<uint, dims> _strides;

public:

    layout() = default;

    layout(std::initializer_list<uint> shape)
    {
        if (shape.size() != dims) throw std::runtime_error("layout::layout(): incorrect rank");
        std::copy(shape.begin(), shape.end(), _shape.begin());
        // compute strides
        for (uint i = 0; i < dims; i++)
        {
            uint stride = 1;
            for (uint j = dims - 1; j > i; j--)
                stride *= _shape[j];
            _strides[i] = stride;
        }
    }

    template<class ... args_type, class = std::enable_if_t<dims == sizeof...(args_type)>>
    explicit layout(args_type&& ... dimension) :
        layout(std::initializer_list<uint>{ (uint)dimension... })
    {}

    constexpr uint shape(uint i) const { return _shape[i]; }
    constexpr uint stride(uint i) const { return _strides[i]; }

    const uint* shape() const { return &_shape.front(); }
    const uint* strides() const { return &_strides.front(); }

    uint total_size() const { return _shape[0] * _strides[0]; }

    layout<dims> reversed() const
    {
        layout<dims> o(*this);
        std::reverse(o._shape.begin(), o._shape.end());
        std::reverse(o._strides.begin(), o._strides.end());
        return o;
    }
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<uint rank>
class tensor
{
    scalar* _ptr;
    layout<rank> _layout;

public:

    inline tensor(scalar* ptr, std::initializer_list<uint> s) :
        _ptr(ptr), _layout(s)
    {}

    inline tensor(scalar* ptr, const layout<2>& s) :
        _ptr(ptr), _layout(s)
    {}

    const layout<rank>& layout() const { return _layout; }

    operator tensor_slice<rank>() const { return tensor_slice<rank>(_ptr, _layout.shape(), _layout.strides()); }

    scalar* ptr() const { return _ptr; }
    uint size() const { return _layout.shape(0); }
    uint total_size() const { return _layout.shape(0) * _layout.stride(0); }
    uint shape(uint i) const { return _layout.shape(i); }
    uint stride(uint i) const { return _layout.stride(i); }

    operator slice() const { return slice(_ptr, total_size()); }

    std::conditional_t<rank == 1, scalar&, tensor_slice<rank - 1>> operator[](uint i) const
    {
        if constexpr (rank == 1)
        {
            return _ptr[i];
        }
        else
        {
            return tensor_slice<rank - 1>(_ptr + (i * _layout.stride(0)), _layout.shape() + 1, _layout.strides() + 1);
        }
    }
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct device_deleter
{
    void operator()(byte* ptr) const noexcept;
};

using device_ptr = std::unique_ptr<byte, device_deleter>;

void device_deleter::operator()(byte* ptr) const noexcept
{
    check(cudaFree(ptr));
}


class pool_allocator
{
    struct chunk
    {
        device_ptr base;
        size_t     next = 0;
        size_t     size = 0;

        chunk() = default;
        chunk(chunk&&) = default;
        chunk(const chunk&) = delete;

        chunk(size_t size);
    };

    void collapse_chunks();
    bool can_allocate(const chunk& ch, size_t alloc_size) const;

    std::vector<chunk> _chunks;

public:

    byte* alloc_bytes(size_t size);

    template<typename type>
    type* alloc_array(size_t length)
    {
        return (type*)alloc_bytes(length * sizeof(type));
    }

    template<uint n>
    tensor<n> alloc(std::initializer_list<uint> shape)
    {
        uint size = 1;
        for (uint dim : shape)
            size *= dim;

        auto ptr = (scalar*)alloc_bytes(sizeof(scalar) * size);
        return tensor<n>(ptr, shape);
    }

    template<class ... args_t, uint n = sizeof...(args_t)>
    tensor<n> alloc(args_t ... shape) { return alloc<n>(std::initializer_list<uint>({ (uint)shape... })); }

    void clear()
    {
        collapse_chunks();
    }
};

pool_allocator::chunk::chunk(size_t size)
{
    byte* ptr = nullptr;
    check(cudaMallocManaged(&ptr, size));

    this->base = device_ptr(ptr);
    this->next = 0;
    this->size = size;
}

bool pool_allocator::can_allocate(const pool_allocator::chunk& c, size_t alloc_size) const
{
    return (c.next + alloc_size) <= c.size;
}

byte* pool_allocator::alloc_bytes(size_t size)
{
    // if new allocation doesn't fit in current chunk then we add a new chunk
    if (_chunks.empty() || !can_allocate(_chunks.back(), size))
    {
        _chunks.emplace_back(size);
    }

    chunk& c = _chunks.back();
    byte* ptr = c.base.get() + c.next;
    c.next += size;
    return ptr;
}

void pool_allocator::collapse_chunks()
{
    if (_chunks.empty())
    {
        return;
    }
    else if (_chunks.size() == 1)
    {
        _chunks.back().next = 0;
    }
    else
    {
        size_t total_sz = 0;
        for (const auto& c : _chunks)
            total_sz += c.next; //used space

        _chunks.clear();
        _chunks.emplace_back(total_sz);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<scalar(*func)(scalar)>
__global__ void map_kernel(const scalar* input, scalar* output)
{
    unsigned int i = (blockDim.x * blockIdx.x) + threadIdx.x;
    output[i] = func(input[i]);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <random>

__global__ void rng_init(curandState* states, unsigned long seed, uint n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
    {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void rng_gen_uniform(curandState* states, float* x, uint n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
    {
        curandState* local_state = &states[idx];
        x[idx] = curand_uniform(local_state);
    }
}

__global__ void rng_gen_normal(curandState* states, float* x, uint n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
    {
        curandState* local_state = &states[idx];
        x[idx] = curand_normal(local_state);
    }
}


class random_generator
{
    pool_allocator _allocator;

public:

    void random_uniform(slice& x);
    void random_normal(slice& x);
};

void random_generator::random_uniform(slice& x)
{
    std::random_device dev;
    curandState* states = _allocator.alloc_array<curandState>(x.size());

    int blockSize = 256;
    int blockCount = (x.size() + blockSize - 1) / blockSize;
    rng_init<<<blockCount, blockSize>>>(states, dev(), x.size());

    rng_gen_uniform<<<blockCount, blockSize>>>(states, x.ptr(), x.size());

    _allocator.clear();
}

void random_generator::random_normal(slice& x)
{
    std::random_device dev;
    curandState* states = _allocator.alloc_array<curandState>(x.size());

    int blockSize = 256;
    int blockCount = (x.size() + blockSize - 1) / blockSize;
    rng_init<<<blockCount, blockSize>>>(states, dev(), x.size());

    rng_gen_normal<<<blockCount, blockSize>>>(states, x.ptr(), x.size());

    _allocator.clear();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class context
{
    pool_allocator _pool;
    random_generator _rng;
    uint _batches = 1;

public:

    void zero(slice x);
    void fill(slice x, scalar value);
    void random_uniform(slice x) { _rng.random_uniform(x); }
    void random_normal(slice x) { _rng.random_normal(x); }

    template<typename ... args_t, uint n = sizeof...(args_t)>
    tensor<n + 1> batch_alloc(args_t ... args) { return alloc(_batches, args...); }

    template<typename ... args_t, uint n = sizeof...(args_t)>
    tensor<n> alloc(args_t ... args) { return _pool.alloc<n>({ (uint)args... }); }

    void sync();
    void clear_allocator() { _pool.clear(); }
};

void context::sync()
{
    check(cudaDeviceSynchronize());
}

void context::zero(slice x)
{
    check(cudaMemset(x.ptr(), 0, x.size() * sizeof(scalar)));
}

__global__ void fill_kernel(scalar* ptr, uint n, scalar val)
{
    uint i = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (i < n) ptr[i] = val;
}

void context::fill(slice x, scalar value)
{
    uint blockSize = 256;
    uint blockCount = (x.size() + blockSize - 1) / blockSize;
    fill_kernel<<<1, x.size()>>>(x.ptr(), x.size(), value);
    sync();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void print(scalar t, uint indent = 0)
{
    //std::string space((indent * 2) + 1, ' ');
    std::cout << t << ", ";
}

template<typename array>
void print(const array& t, uint indent = 0)
{
    std::string space((indent * 2) + 1, ' ');

    std::cout << "\n";

    std::cout << space << "[";
    for (uint i = 0; i < t.size(); i++)
    {
        print(t[i], indent + 1);
    }
    std::cout << space << "],\n";
}


static_assert(std::is_trivially_destructible_v<slice>, "slice is not trivially destructible");
static_assert(std::is_trivially_destructible_v<tensor_slice<2>>, "slice is not trivially destructible");

/*
__global__ void mul(tensor<2> m, tensor<1> a, tensor<1> o)
{
    unsigned int row = (blockDim.x * blockIdx.x) + threadIdx.x;
    for (unsigned int col = 0; col < a.size(); col++)
    {
        o[row] = m[row][col] * a[col];
    }
}
*/

__global__ void mul_kernel(const scalar* mat, uint rows, uint cols, const scalar* in, scalar* out)
{
    unsigned int row = (blockDim.x * blockIdx.x) + threadIdx.x;
    scalar dot = 0;
    for (unsigned int col = 0; col < cols; col++)
    {
        dot += mat[(cols * row) + col] * in[col];
    }
    out[row] = dot;
}

void mul(const tensor<2>& mat, const slice& in, slice& out)
{
    mul_kernel<<<1, mat.shape(0)>>>(mat.ptr(), mat.shape(0), mat.shape(1), in.ptr(), out.ptr());
}

__device__ float multiply_by_2(float x)
{
    return x * 2;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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


__global__ void matrix_set_rows_kernel(uint rows, uint cols, scalar* m, const scalar* v)
{
    uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint j = (blockIdx.y * blockDim.y) + threadIdx.y;
    if ((i < cols) && (j < rows))
    {
        m[(j * cols) + i] = v[i];
    }
}

void matrix_set_rows(tensor<2>& m, const tensor<1>& a)
{
    ASSERT(m.shape(1) == a.size());

    dim3 block_size = dim3(32, 32);
    dim3 block_count = dim3(
        (m.shape(1) + block_size.x - 1) / block_size.x, //i - col
        (m.shape(0) + block_size.y - 1) / block_size.y  //j - row
    );
    matrix_set_rows_kernel<<<block_count, block_size>>>(m.shape(0), m.shape(1), m.ptr(), a.ptr());

    //for (uint row = 0; row < m.shape(0); row++)
    //	check(cudaMemcpyAsync(m[row].ptr(), a.ptr(), a.size() * sizeof(scalar), cudaMemcpyDeviceToDevice));
}

tensor<2> transpose(const tensor<2>& x)
{
    return tensor<2>(x.ptr(), x.layout().reversed());
}

enum matrix_flag
{
    transpose_A = 1,
    transpose_B = 2,
    accumulate  = 4
};

// F = DE
// C = AB
void matrix_mul(tensor<2>& c, tensor<2> a, const tensor<2>& b, matrix_flag flags = matrix_flag(0))
{
    bool trn_A = flags & matrix_flag::transpose_A;
    bool trn_B = flags & matrix_flag::transpose_B;

    auto opA = trn_A ? CUBLAS_OP_T : CUBLAS_OP_N;
    auto opB = trn_B ? CUBLAS_OP_T : CUBLAS_OP_N;

    uint colsA = trn_A ? a.shape(0) : a.shape(1);
    uint rowsA = trn_A ? a.shape(1) : a.shape(0);
    uint colsB = trn_B ? b.shape(0) : b.shape(1);
    uint rowsB = trn_B ? b.shape(1) : b.shape(0);

    // check dimensions
    ASSERT(colsA == rowsB);
    ASSERT(c.shape(0) == rowsA);
    ASSERT(c.shape(1) == colsB);

    uint m = colsB;  //number of rows of matrix op(A) and C.           == cols of B
    uint n = rowsA;  //number of columns of matrix op(B) and C.        == rows of A
    uint k = colsA;  //number of columns of op(A) and rows of op(B).   == cols of A

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

void matrix_sum_rows(context& dc, tensor<1>& sum, tensor<2>& m)
{
    ASSERT(sum.size() == m.shape(1));

    auto temp = dc.alloc(m.shape(0));
    dc.fill(temp, 1.0f);

    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSgemv(get_handle(), CUBLAS_OP_N, m.shape(1), m.shape(0),
        &alpha, m.ptr(), m.shape(1), temp.ptr(), 1, &beta, sum.ptr(), 1);
}

scalar vector_sum(const tensor<1>& a)
{
    scalar r = 0;
    check(cublasSasum(get_handle(), a.size(), a.ptr(), 1, &r));
    return r;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void test_matrix_mul(context& dc)
{
    std::cout << "test matrix mul:\n";

    auto m = dc.alloc(3, 6);
    auto a = dc.alloc(6, 2);
    auto c = dc.alloc(3, 2);
    dc.sync();
    std::vector<scalar> mbuf = {
        2,   3,  4,  5,  6,  7,
        8,   9, 10, 11, 12, 13,
        -6, -5, -4, -3, -2, -1
    };
    std::copy(mbuf.begin(), mbuf.end(), m.ptr());
    std::vector<scalar> abuf = {
         1, -2,
         3, -3,
         5, -4,
         7, -5,
         9, -6,
        11, -7
    };
    std::copy(abuf.begin(), abuf.end(), a.ptr());
    dc.sync();

    matrix_mul(c, m, a);
    dc.sync();

    print(m);
    print(a);
    print(c);

    ASSERT(c[0][0] == 197);
    ASSERT(c[0][1] == -139);
    ASSERT(c[1][0] == 413);
    ASSERT(c[1][1] == -301);
    ASSERT(c[2][0] == -91);
    ASSERT(c[2][1] == 77);
}


void test_matrix_mul_transpose(context& dc)
{
    std::cout << "test transposed matrix mul:\n";

    auto m = dc.alloc(3, 6);
    auto a = dc.alloc(3, 1);
    auto c = dc.alloc(6, 1);
    dc.sync();
    std::vector<scalar> mbuf = {
        2,   3,  4,  5,  6,  7,
        8,   9, 10, 11, 12, 13,
        -6, -5, -4, -3, -2, -1
    };
    std::copy(mbuf.begin(), mbuf.end(), m.ptr());
    std::vector<scalar> abuf = { 5, 8, 10 };
    std::copy(abuf.begin(), abuf.end(), a.ptr());
    dc.sync();

    matrix_mul(c, m, a, matrix_flag::transpose_A);
    dc.sync();

    print(m);
    print(a);
    print(c);

    ASSERT(c[0][0] == 14);
    ASSERT(c[1][0] == 37);
    ASSERT(c[2][0] == 60);
    ASSERT(c[3][0] == 83);
    ASSERT(c[4][0] == 106);
    ASSERT(c[5][0] == 129);
}

void test_matrix_mul_transpose2(context& dc)
{
    std::cout << "test transposed matrix mul 2:\n";

    auto u = dc.alloc(3, 2);
    auto v = dc.alloc(4, 2);
    auto c = dc.alloc(3, 4);

    dc.sync();
    std::vector<scalar> buf;

    buf = {
        1, 2,
        3, 4,
        5, 6
    };
    std::copy(buf.begin(), buf.end(), u.ptr());

    buf = {
        8, 7,
        6, 5,
        4, 3,
        2, 1
    };
    std::copy(buf.begin(), buf.end(), v.ptr());
    dc.sync();

    matrix_mul(c, u, v, matrix_flag::transpose_B);
    dc.sync();

    print(c);
}

void test_matrix(context& dc)
{
    test_matrix_mul(dc);
    test_matrix_mul_transpose(dc);
    test_matrix_mul_transpose2(dc);

    auto mat = dc.alloc(4, 3);
    auto x = dc.alloc(2, 4);
    auto y = dc.alloc(2, 3);
    auto b = dc.alloc(3);

    dc.zero(b);
    b[0] = 1.1f; b[1] = 2.1f; b[2] = 3.1f;

    /*
    //test sum rows
    auto sum = dc.alloc(b.size());
    matrix_set_rows(mat, b);
    matrix_sum_rows(dc, sum, mat);
    dc.sync();
    print(sum);
    return;
    */

    dc.fill(mat, 1.0f);
    dc.fill(x, 1.0f);

    matrix_set_rows(y, b);
    matrix_mul(y, x, mat, matrix_flag::accumulate);

    dc.sync();

    std::cout << "mat:\n";
    print(mat);
    std::cout << "x:\n";
    print(x);
    std::cout << "y:\n";
    print(y);

    /*
    auto mat2 = ctx.alloc(10, 10);
    ctx.random_uniform(slice(mat2.ptr(), mat2.total_size()));
    ctx.sync();
    print(mat2);
    */

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


__global__ void fill_test(uint n, float* ptr)
{
    auto i = global_index();
    ptr[i] = 24;
    
}

template<typename ... args_t>
using device_kernel = void(*)(args_t...);

void test_launch(context& dc, device_kernel<uint, float*> x)
{
    auto p  =dc.alloc(100);
    x<<<1,100>>>(100, p.ptr());
    cudaDeviceSynchronize();
    print(p);
}

int main()
{
    context ctx;

    test_launch(ctx, &fill_test);
    
    test_matrix(ctx);
    return 0;

    auto mat = ctx.alloc(10, 10);
    auto x = ctx.alloc(10);
    auto y = ctx.alloc(10);
    ctx.sync();

    for (uint j = 0; j < mat.shape(0); j++)
        mat[j][j] = 1.0f;

    ctx.fill(slice(x), 5.0f);
    ctx.zero(slice(y));
    ctx.sync();

    //times2<<<1, vec.size()>>>(slice(vec));
    //times2<<<1,1000>>>(&vec[0], vec.size());
    
    if (0)
    {
        map_kernel<&multiply_by_2><<<1, x.size()>>>(x.ptr(), x.ptr());
        ctx.sync();

        float e = 0.0f;
        for (uint i = 0; i < x.size(); i++)
            e += std::abs(10.0f - x[i]);
        std::cout << e << std::endl;
    }

    uint blockSize = 256;
    uint blockCount = (x.size() + blockSize - 1) / blockSize;

    //tensor_slice<2> _mat = mat;
    //tensor_slice<1> _vec = vec;
    //tensor_slice<1> _out = out;
    //mul<<<1, 1000>>>(mat, vec, out);
    mul(mat, x, slice(y));
    ctx.sync();

    float error = 0.0f;
    for (uint i = 0; i < y.size(); i++)
        error += std::abs(5.0f - y[i]);

    std::cout << "error: " << error << std::endl;

    return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
