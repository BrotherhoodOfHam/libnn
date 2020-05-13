/*
    Random number generator
*/

#include <random>

#include "gpu.h"
#include "nn/device.h"

#include <curand.h>
#include <curand_kernel.h>

using namespace nn;

/*************************************************************************************************************************************/

// x is an array of uniformly distributed random numbers that we will transform
__global__ void RNG_bernoulli_kernel(float* x, uint n, float p, float v)
{
    uint i = global_index();
    if (i < n)
    {
        x[i] = (x[i] < p) ? v : 0.0f;
    }
}

/*************************************************************************************************************************************/

rng::rng() :
    rng(std::random_device()())
{}

rng::~rng()
{
    check(curandDestroyGenerator(_prng));
}

rng::rng(seed_type seed)
{
    check(curandCreateGenerator(&_prng, CURAND_RNG_PSEUDO_DEFAULT));
    check(curandSetPseudoRandomGeneratorSeed(_prng, seed));
}

void rng::seed(seed_type seed)
{
    check(curandSetPseudoRandomGeneratorSeed(_prng, seed));
}

void rng::random_uniform(vector x)
{
    check(curandGenerateUniform(_prng, x.ptr(), x.size()));
}

void rng::random_normal(vector x, float sdv, float mean)
{
    check(curandGenerateNormal(_prng, x.ptr(), x.size(), mean, sdv));
}

void rng::random_bernoulli(vector x, float probability, float value)
{
    check(curandGenerateUniform(_prng, x.ptr(), x.size()));
    
    uint block_size = 256;
    uint block_count = ((uint)x.size() + block_size - 1) / block_size;

    RNG_bernoulli_kernel<<<block_count, block_size>>>(x.ptr(), x.size(), probability, value);
}

/*************************************************************************************************************************************/
