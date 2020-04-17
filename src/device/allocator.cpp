#include "nn/device.h"
#include "kernels.h"

using namespace nn;

template<typename ... fmt_args>
void debug_printf(const char* fmt, fmt_args&& ... args)
{
    if (context::is_debug())
    {
        printf(fmt, args...);
    }
}

/*************************************************************************************************************************************/

buffer::buffer(uint size) :
    _size(size)
{
    scalar* ptr;
    check(cudaMalloc(&ptr, sizeof(scalar) * size));

    debug_printf("[buffer] cudaMalloc: 0x%p  %dB\n", ptr, _size);

    _ptr.reset(ptr, [](scalar* ptr) {
        if (ptr != nullptr)
        {
            debug_printf("[buffer] cudaFree: 0x%p\n", ptr);

            check(cudaFree(ptr));
        }
    });
}

/*************************************************************************************************************************************/

void device_deleter::operator()(byte* ptr) const noexcept
{
    if (ptr != nullptr)
    {
        debug_printf("[device_deleter] cudaFree: 0x%p\n", ptr);

        check(cudaFree(ptr));
    }
}

pool_allocator::chunk::chunk(size_t size)
{
    byte* ptr = nullptr;
    check(cudaMalloc(&ptr, size));

    debug_printf("[pool_allocator] cudaMalloc: 0x%p  %dB\n", ptr, size);

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

/*************************************************************************************************************************************/
