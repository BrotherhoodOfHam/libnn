/*
    GPU header

    contains definitions for CUDA and some helper functions
*/

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <exception>
#include <string>

#include "nn/tensors.h"
#include "nn/device.h"

namespace nn
{
#ifdef __CUDACC__
    __device__ inline uint global_index()
    {
        return (blockIdx.x * blockDim.x) + threadIdx.x;
    }
#else
    inline uint global_index() { return -1; }
#endif

    /*
    template<scalar(*func)(scalar)>
    __global__ void map_kernel(uint n, const scalar* input, scalar* output)
    {
        uint i = global_index();
        if (i < n)
        {
            output[i] = func(input[i]);
        }
    }

    template<scalar(*func)(scalar)>
    inline vector map_vector(scope& dc, const vector& x)
    {
        auto y = dc.alloc(x.size());

        uint block_size = 256;
        uint block_count = (x.size() + block_size - 1) / block_size;

        map_kernel<func><<<block_count, block_size>>>(x.size(), x.ptr(), y.ptr());

        return y;
    }
    */

    template<typename error_type>
    inline void check(error_type error)
    {
        if (error)
        {
            std::string msg = "cudaError: " + std::to_string(error);
            std::cerr << msg << std::endl;
            throw std::runtime_error(msg);
        }
    }

    inline void _tensor_print(scalar t, size_t indent = 0)
    {
        std::cout << t << ", ";
    }

    template<typename array>
    inline void _tensor_print(const array& t, size_t indent = 0)
    {
        std::string space((indent * 2) + 1, ' ');

        std::cout << "\n";

        std::cout << space << "[";
        for (uint i = 0; i < t.shape(0); i++)
        {
            _tensor_print(t[i], indent + 1);
        }
        std::cout << space << "],\n";
    }

    template<uint rank>
    inline void debug_print(const tensor<rank>& t)
    {
        check(cudaDeviceSynchronize());
        std::vector<scalar> _buf(t.size());
        check(cudaMemcpy(_buf.data(), t.ptr(), _buf.size() * sizeof(scalar), cudaMemcpyDeviceToHost));

        std::cout << "shape = (";
        uint i = 0;
        for (uint s : t.shape())
        {
            std::cout << s;
            i++;
            if (i < rank)
                std::cout << ", ";
        }
        std::cout << ")\n";

        tensor<rank> host_tensor(_buf.data(), t.layout());
        _tensor_print(host_tensor);
    }

    template<uint rank>
    inline void debug_print(const char* msg, const tensor<rank>& t)
    {
        std::cout << msg << std::endl;
        debug_print(t);
    }

    template<typename ... fmt_args>
    inline void debug_printf(const char* fmt, fmt_args&& ... args)
    {
        if (device::is_debug())
        {
            printf(fmt, args...);
        }
    }
}
