#include "kernels.h"
#include "nn/device.h"

using namespace nn;

/*************************************************************************************************************************************/

static bool g_debug_mode = false;

void context::set_debug(bool on)
{
    g_debug_mode = on;
}
bool context::is_debug()
{
    return g_debug_mode;
}

/*************************************************************************************************************************************/

__global__ void fill_kernel(scalar* ptr, uint n, scalar val)
{
    uint i = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (i < n) ptr[i] = val;
}

/*************************************************************************************************************************************/

void context::sync() const
{
    check(cudaDeviceSynchronize());
}

void context::zero(vector x) const
{
    check(cudaMemset(x.ptr(), 0, x.size() * sizeof(scalar)));
}

void context::fill(vector x, scalar value) const
{
    uint blockSize = 256;
    uint blockCount = (x.size() + blockSize - 1) / blockSize;
    fill_kernel<<<1, x.size()>>>(x.ptr(), x.size(), value);
    sync();
}

void context::update(vector x, const const_span<scalar>& values) const
{
    assert(x.size() == values.size());

    check(cudaMemcpy(x.ptr(), values.begin(), sizeof(scalar) * values.size(), cudaMemcpyHostToDevice));
}

void context::read(vector x, std::vector<scalar>& out) const
{
    out.resize(x.size());

    check(cudaMemcpy(out.data(), x.ptr(), sizeof(scalar) * x.size(), cudaMemcpyDeviceToHost));
}

/*************************************************************************************************************************************/
