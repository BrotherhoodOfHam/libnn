/*
    Device related classes
*/

#pragma once

#include <memory>

#include "tensors.h"

class curandGenerator_st;

namespace nn
{
    /*************************************************************************************************************************************/

    class device;

    struct device_deleter
    {
        void operator()(byte* ptr) const noexcept;
    };

    using device_ptr = std::unique_ptr<byte, device_deleter>;

    /*
        Simple device buffer
    */
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

    /*
        Simple pool allocator

        Allocations are never freed but instead compacted
    */
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

    /*************************************************************************************************************************************/

    /*
        Accelerated random number generator
    */
    class rng
    {
        curandGenerator_st* _prng;

    public:

        using seed_type = size_t;

        rng();
        rng(seed_type seed);
        ~rng();

        void seed(seed_type seed);

        void random_uniform(vector x);
        void random_normal(vector x, float sdv, float mean);
        void random_bernoulli(vector x, float probability);
    };

    /*
        Stateless functions for manipulating vectors and tensors
    */
    class ops
    {
    public:

        // Flag usage depends on op function
        enum flag
        {
            transpose_A = 1,    // transpose the first arg
            transpose_B = 2,    // transpose the second arg
            accumulate  = 4     // accumulate rather than assign
        };

        // Copying between host and device
        void update(vector x, const const_span<scalar>& values);
        void read(vector x, std::vector<scalar>& out);

        // Basic ops
        void zero(vector x);
        void fill(vector x, scalar value);

        /*
            Linear algebra
        */

        // c[i] = a[i] * b[i]
        void vector_mul(vector& c, const vector& a, const vector& b);
        // c[i] = a[i] + b[i]
        void vector_add(vector& c, const vector& a, const vector& b);
        // c[i] = a[i] - b[i]
        void vector_sub(vector& c, const vector& a, const vector& b);
        // sum += a[i] for all i
        scalar vector_sum(const vector& a);

        // c[i,k] = a[i,j] * b[j,k]
        void matrix_mul(tensor<2>& c, const tensor<2>& a, const tensor<2>& b, flag flags = flag(0));
        // m[k,i] = a[i] for all k
        void matrix_set_rows(tensor<2>& m, const vector& a);
        // sum[i] += m[k,i] for all k
        void matrix_sum_rows(vector& sum, const tensor<2>& m);
    };

    inline ops::flag operator|(ops::flag a, ops::flag b) { return ops::flag((int)a | (int)b); }
    inline bool operator&(ops::flag a, ops::flag b) { return ((int)a & (int)b) != 0; }

    /*************************************************************************************************************************************/

    enum class execution_mode
    {
        execute = 0,
        training = 1
    };

    /*
        Device variable scope.

        Allows temporary memory allocation. When the scope is destroyed the memory should not be accessed.
    */
    class scope : public ops
    {
        device*         _d;
        pool_allocator* _pool;
        uint            _batch_size;
        execution_mode  _mode;

        scope() = default;
        scope(device* d, pool_allocator* pool, execution_mode mode, uint batch_size);

        void check() const;

    public:

        friend class device;

        explicit scope(const scope&) = delete;
        scope(scope&&) noexcept;
        ~scope();

        bool is_training() const { return _mode == execution_mode::training; }
        uint batch_size() const { return _batch_size; }

        // inherit device behaviour
        inline void random_uniform(vector x);
        inline void random_normal(vector x, float sdv = 1.0f, float mean = 0.0f);

        template<uint n>
        tensor<n + 1> to_batched(const vector& x, const tensor_layout<n>& ly)
        {
            return x.reshape(tensor_layout<n + 1>(_batch_size, ly));
        }

        template<uint n>
        tensor<n+1> batch_alloc(const tensor_layout<n>& ly) { return _pool->alloc(tensor_layout<n + 1>(_batch_size, ly)); }

        template<typename ... args_t, uint n = sizeof...(args_t) + 2>
        tensor<n> batch_alloc(uint shape0, args_t ... shape) { return this->alloc(_batch_size, shape0, shape...); }

        template<uint n>
        tensor<n> alloc(const tensor_layout<n>& ly) { return _pool->alloc(ly); }

        template<typename ... args_t, uint n = sizeof...(args_t) + 1>
        tensor<n> alloc(uint shape0, args_t ... shape) { return _pool->alloc<n>(tensor_layout<n>(shape0, (uint)shape...)); }
    };

    /*
        Device class represents the logical device,
        there is one per application
    */
    class device : public ops
    {
        pool_allocator   _scope_pool;
        rng              _rng;
        bool             _debug_mode = false;
        bool             _in_scope   = false;

        device() = default;

    public:

        friend class scope;

        static device& get();

        static void set_debug(bool on) { get()._debug_mode = on; }
        static bool is_debug() { return get()._debug_mode; }

        device(const device&) = delete;
        device(device&&) = delete;

        // Enter a variable scope.
        // There can only be one instance at a time
        scope scope(execution_mode mode, uint batch_size);

        void random_uniform(vector x) { _rng.random_uniform(x); }
        void random_normal(vector x, float sdv = 1.0f, float mean = 0.0f) { _rng.random_normal(x, sdv, mean); }
    };

    inline void scope::random_uniform(vector x) { check(); _d->random_uniform(x); }
    inline void scope::random_normal(vector x, float sdv, float mean) { check(); _d->random_normal(x, sdv, mean); }

    /*************************************************************************************************************************************/
}
