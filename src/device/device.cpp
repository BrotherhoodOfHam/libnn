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

scope device::scope(execution_mode mode, uint batch_size)
{
    if (_in_scope)
    {
        throw std::runtime_error("The scope cannot be aquired");
    }
    _in_scope = true;
    return nn::scope(this, &_scope_pool, mode, batch_size);
}

/*************************************************************************************************************************************/
