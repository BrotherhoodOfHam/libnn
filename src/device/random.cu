/*
    Random number generator
*/

#include <random>

#include "kernels.h"
#include "nn/device.h"

#include <curand.h>
#include <curand_kernel.h>

using namespace nn;

/*************************************************************************************************************************************/

__global__ void RNG_bernoulli_kernel(float* x, uint n, random_generator::seed_type seed, float p)
{
    uint i = global_index();
    if (i < n)
    {
        curandState local_state;
        curand_init(seed, threadIdx.x, 0, &local_state);

        x[i] = (curand_uniform(&local_state) < p) ? 1.0f : 0.0f;
    }
}

/*************************************************************************************************************************************/

random_generator::random_generator() :
    random_generator(std::random_device()())
{}

random_generator::~random_generator()
{
    check(curandDestroyGenerator(_prng));
}

random_generator::random_generator(seed_type seed)
{
    check(curandCreateGenerator(&_prng, CURAND_RNG_PSEUDO_DEFAULT));
    check(curandSetPseudoRandomGeneratorSeed(_prng, seed));
}

/*
void random_generator::init(seed_type seed, size_t size)
{
    bool reseed = seed != _seed;
    if (size > _states_size)
    {
        _allocator.free();
        _states = _allocator.alloc_array<curandState>(size);
        _states_size = size;
        reseed = true;
    }

    if (reseed)
    {
        _seed = seed;

        uint block_size = 256;
        uint block_count = ((uint)size + block_size - 1) / block_size;
        RNG_init_kernel<<<block_count, block_size>>>((curandState*)_states, _seed, (uint)size);
    }
}
*/

void random_generator::seed(seed_type seed)
{
    check(curandSetPseudoRandomGeneratorSeed(_prng, seed));
}

void random_generator::random_uniform(vector x)
{
    check(curandGenerateUniform(_prng, x.ptr(), x.size()));
}

void random_generator::random_normal(vector x, float sdv, float mean)
{
    check(curandGenerateNormal(_prng, x.ptr(), x.size(), mean, sdv));
}

void random_generator::random_bernoulli(vector x, float probability)
{
    uint block_size = 256;
    uint block_count = ((uint)x.size() + block_size - 1) / block_size;

    //curandGenerateBinomial(_prng, x.ptr(), x.size(), 1u, (double)probability);

    seed_type seed = std::random_device()();
    RNG_bernoulli_kernel<<<block_count, block_size>>>(x.ptr(), x.size(), seed, probability);
}

/*************************************************************************************************************************************/
