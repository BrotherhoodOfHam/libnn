/*
    Device related classes
*/

#pragma once

#include <memory>

#include "tensors.h"

class curandGenerator_st;

namespace nn
{
    struct device_deleter
    {
        void operator()(byte* ptr) const noexcept;
    };

    using device_ptr = std::unique_ptr<byte, device_deleter>;

    class buffer
    {
        std::shared_ptr<scalar> _ptr;
        uint                    _size = 0;

    public:

        buffer() = default;
        buffer(buffer&& rhs) = default;
        buffer(const buffer&) = default;

        explicit buffer(uint size);

        scalar* ptr() const { return _ptr.get(); }
        uint size() const { return _size; }
        bool is_empty() const { return _size == 0; }

        template<uint dims>
        tensor<dims> as_tensor(const tensor_layout<dims>& l) const { assert(l.total_size() == _size);  return tensor<dims>(_ptr.get(), l); }
        vector as_vector() const { return vector(_ptr.get(), tensor_layout<1>(_size)); }
    };


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
        tensor<n> alloc(const tensor_layout<n>& shape)
        {
            scalar* ptr = (scalar*)alloc_bytes(sizeof(scalar) * shape.total_size());
            return tensor<n>(ptr, shape);
        }

        void free()
        {
            collapse_chunks();
        }
    };

    /*
        Accelerated random number generator
    */
    class random_generator
    {
        curandGenerator_st* _prng;

    public:

        using seed_type = size_t;

        random_generator();
        random_generator(seed_type seed);
        ~random_generator();

        void seed(seed_type seed);

        void random_uniform(vector x);
        void random_normal(vector x, float sdv, float mean);
        void random_bernoulli(vector x, float probability);
    };

    enum class execution_mode
    {
        execute  = 0,
        training = 1
    };

    /*
        Device context
    */
    class context
    {
        pool_allocator           _pool;
        mutable random_generator _rng;
        uint                     _batch_size = 1;
        execution_mode           _mode = execution_mode::execute;

    public:

        context() = default;
        context(const context&) = delete;

        static const context& get_global()
        {
            static context dc;
            return dc;
        }

        static void set_debug(bool on);
        static bool is_debug();

        void set_mode(execution_mode mode) { _mode = mode; }
        void set_batches(uint batches) { _batch_size = batches; }
        bool is_training() const { return _mode == execution_mode::training; }
        uint batch_size() const { return _batch_size; }

        void zero(vector x) const;
        void fill(vector x, scalar value) const;
        void random_uniform(vector x) const { _rng.random_uniform(x); }
        void random_normal(vector x, float sdv = 1.0f, float mean = 0.0f) const { _rng.random_normal(x, sdv, mean); }
        void update(vector x, const const_span<scalar>& values) const;
        void read(vector x, std::vector<scalar>& out) const;

        template<uint n>
        tensor<n + 1> to_batched(const vector& x, const tensor_layout<n>& ly)
        {
            return x.reshape(tensor_layout<n + 1>(_batch_size, ly));
        }

        template<uint n>
        tensor<n+1> batch_alloc(const tensor_layout<n>& ly) { return _pool.alloc(tensor_layout<n + 1>(_batch_size, ly)); }

        template<typename ... args_t, uint n = sizeof...(args_t) + 2>
        tensor<n> batch_alloc(uint shape0, args_t ... shape) { return this->alloc(_batch_size, shape0, shape...); }

        template<uint n>
        tensor<n> alloc(const tensor_layout<n>& ly) { return _pool.alloc(ly); }

        template<typename ... args_t, uint n = sizeof...(args_t) + 1>
        tensor<n> alloc(uint shape0, args_t ... shape) { return _pool.alloc<n>(tensor_layout<n>(shape0, (uint)shape...)); }

        void sync() const;
        void clear_allocator() { _pool.free(); }
    };
}
