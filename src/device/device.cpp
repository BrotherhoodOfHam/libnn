/*
    Device source
*/

#include "gpu.h"
#include "nn/device.h"

using namespace nn;

// device is a singleton
device& device::get()
{
    static device d;
    return d;
}

/*************************************************************************************************************************************
    Variable scope
*************************************************************************************************************************************/

scope::scope(device* d, pool_allocator* pool, execution_mode mode, uint batch_size) :
    _d(d), _pool(pool), _mode(mode), _batch_size(batch_size)
{}

scope::scope(scope&& rhs) noexcept :
    _d(rhs._d), _pool(rhs._pool), _mode(rhs._mode), _batch_size(rhs._batch_size)
{
    rhs._d = nullptr;
    rhs._pool = nullptr;
    rhs._batch_size = 0;
}

scope::~scope()
{
    // empty the pool
    if (_pool != nullptr)
        _pool->free();
    // report scope has been released
    if (_d != nullptr)
        _d->_in_scope = false;
}

void scope::check() const
{
    assert(_d != nullptr);
}

scope device::begin(execution_mode mode, uint batch_size)
{
    auto& d = get();
    if (d._in_scope)
    {
        throw std::runtime_error("The scope cannot be aquired");
    }
    d._in_scope = true;
    return nn::scope(&d, &d._scope_pool, mode, batch_size);
}

/*************************************************************************************************************************************/

device_timer::device_timer()
{
    check(cudaEventCreate((cudaEvent_t*)&_start));
    check(cudaEventCreate((cudaEvent_t*)&_stop));
}

device_timer::~device_timer()
{
    check(cudaEventDestroy((cudaEvent_t)_start));
    check(cudaEventDestroy((cudaEvent_t)_stop));
}

void device_timer::start()
{
    check(cudaEventRecord((cudaEvent_t)_start));
}

float device_timer::stop()
{
    check(cudaEventRecord((cudaEvent_t)_stop));

    check(cudaEventSynchronize((cudaEvent_t)_stop));
    
    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, (cudaEvent_t)_start, (cudaEvent_t)_stop));

    check(cudaDeviceSynchronize());

    return ms;
}

/*************************************************************************************************************************************/
