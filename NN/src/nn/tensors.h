/*
	Tensor classes.
*/

#pragma once

#include <array>
#include <numeric>

#include "common.h"

/*************************************************************************************************************************************/

namespace nn
{
	enum tensor_format
	{
		NW    = 2,
		NCW   = 3,
		NCHW  = 4,
	};

	typedef float scalar;

	/*
		Tensor shape is a tuple of integer values storing the extent of each dimension
	*/
	class tensor_shape
	{
	private:

		static constexpr size_t s_capacity = 4;

		uint* _shape;
		uint _length;
		std::array<uint, s_capacity> _storage;

	public:

		using value_type = uint;
		using reference = const uint&;
		using const_reference = const uint&;
		using size_type = uint;

		using iterator = const uint*;
		using const_iterator = const uint*;

		tensor_shape()
		{
			_length = 0;
			_shape = _storage.data();
		}

		tensor_shape(std::initializer_list<uint> shape)
		{
			_length = shape.size();
			_shape = (_length > s_capacity) ? new uint[_length] : _storage.data();
			std::copy(shape.begin(), shape.end(), _shape);
		}

		template<class ... args_type>
		tensor_shape(args_type ... shape)
		{
			std::initializer_list<uint> i = { (uint)shape... };
			this->tensor_shape::tensor_shape(i);
		}

		tensor_shape(const tensor_shape& rhs)
		{
			_length = rhs.length();
			_shape = (_length > s_capacity) ? new uint[_length] : _storage.data();
			std::copy(rhs.begin(), rhs.end(), _shape);
		}

		void operator=(const tensor_shape& rhs)
		{
			if (_length > s_capacity) delete _shape;
			_length = rhs.length();
			_shape = (_length > s_capacity) ? new uint[_length] : _storage.data();
			std::copy(rhs.begin(), rhs.end(), _shape);
		}

		~tensor_shape()
		{
			if (_length > s_capacity)
				delete _shape;
		}

		inline iterator begin() const { return _shape; }
		inline iterator end() const { return _shape + _length; }
		inline uint length() const { return _length; }

		inline uint memory_size() const
		{
			return std::accumulate(_shape, _shape + _length, 1, std::multiplies<size_t>());
		}

		inline uint operator[](uint i) const
		{
			assert(i < _length);
			return _shape[i];
		}

		inline uint& operator[](uint i)
		{
			assert(i < _length);
			return _shape[i];
		}
	};

	/*
		N dimensional tensor.
	*/
	class tensor
	{
		std::vector<scalar> _data;
		tensor_shape        _shape;     // bounds of data
		tensor_shape        _strides;   // stride of data (for indexing)

	public:

		tensor(const tensor&) = delete;

		template<typename ... args_t>
		tensor(args_t ... shape) :
			tensor(tensor_shape(shape...))
		{}

		tensor(const tensor_shape& shape) :
			_data(shape.memory_size()),
			_shape(shape)
		{
			compute_strides();
		}

		tensor(tensor&& rhs) noexcept :
			_data(std::move(rhs._data)),
			_shape(std::move(rhs._shape)),
			_strides(std::move(rhs._strides))
		{}

		// Properties
		inline const tensor_shape& shape() const { return _shape; }
		inline const tensor_shape& stride() const { return _strides; }
		inline uint shape(uint i) const { return _shape[i]; }

		// Get element at coordinate in tensor
		template<class ... args_t, uint n = sizeof...(args_t)>
		inline scalar& at(args_t ... index)
		{
			assert(_shape.length() == n);
			std::array<uint, n> indices = { (uint)index... };
			auto ptr = _data.data();
			for (uint i = 0; i < n; i++)
				ptr += indices[i] * _strides[i];
			return *ptr;
		}
		template<class ... args_t, uint n = sizeof...(args_t)>
		inline scalar at(args_t ... index) const
		{
			assert(_shape.length() == n);
			std::array<uint, n> indices = { (uint)index... };
			auto ptr = _data.data();
			for (uint i = 0; i < n; i++)
				ptr += indices[i] * _strides[i];
			return *ptr;
		}

		template<class ... args_t, uint n = sizeof...(args_t)>
		inline scalar& operator()(args_t ... index) { return at(index...); }
		template<class ... args_t, uint n = sizeof...(args_t)>
		inline scalar operator()(args_t ... index) const { return at(index...); }

		// check if two tensor shapes are equivalent
		static void check(const tensor_shape& lhs, const tensor_shape& rhs)
		{
			assert(lhs.length() == rhs.length());

			for (uint i = 0; i < lhs.length(); i++)
			{
				assert(lhs[i] == rhs[i]);
			}
		}
		static void check(const tensor& lhs, const tensor& rhs) { return check(lhs.shape(), rhs.shape()); }

		// Get element at 1D index
		inline scalar at_index(uint i) const { return _data[i]; }
		inline scalar& at_index(uint i) { return _data[i]; }
		inline size_t memory_size() const { return _data.size(); }

	private:

		void compute_strides()
		{
			_strides = _shape;
			for (size_t i = 0; i < _shape.length(); i++)
			{
				size_t stride = 1;
				for (size_t j = _shape.length() - 1; j > i; j--)
					stride *= _shape[j];
				_strides[i] = stride;
			}
		}
	};
}

/*************************************************************************************************************************************/
