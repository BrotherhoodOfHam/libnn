/*
	Tensor classes.
*/

#pragma once

#include <array>
#include <numeric>
#include <algorithm>
#include <execution>

#include "common.h"

namespace nn
{
	/*************************************************************************************************************************************/

	template<uint dims>
	class layout;

	template<uint dims>
	class tensor_slice;

	class vector_slice;

	using scalar = float;

	class buffer
	{
		std::shared_ptr<scalar[]> _ptr;
		uint                      _size = 0;

	public:

		buffer() = default;
		buffer(buffer && rhs) = default;
		buffer(const buffer&) = default;

		explicit buffer(uint size) :
			_size(size),
			_ptr(new scalar[size])
		{}

		scalar* ptr() const { return _ptr.get(); }
		uint size() const { return _size; }
		bool is_empty() const { return _size == 0; }

		template<uint dims>
		tensor_slice<dims> as_tensor(const layout<dims>&) const;
		vector_slice as_vector() const;
	};

	class extents : public span<const uint>
	{
	public:
		
		extents(const std::vector<uint>& shape) :
			span(&shape[0], &shape[0] + shape.size())
		{}

		template<size_t dims>
		extents(const std::array<uint, dims>& shape) :
			span(&shape[0], &shape[0] + shape.size())
		{}

		extents(std::initializer_list<uint> shape) :
			span(shape.begin(), shape.end())
		{}

		extents(const uint* begin, const uint* end) :
			span(begin, end)
		{}

		operator std::vector<uint>() { return std::vector<uint>(begin(), end()); }

		uint total() const
		{
			uint n = 1;
			for (uint i : *this)
				n *= i;
			return n;
		}

		static bool equals(const extents& lhs, const extents& rhs)
		{
			return std::equal(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
		}
	};


	template<uint dims>
	class layout
	{
		std::array<uint, dims> _shape;
		std::array<uint, dims> _strides;
		uint _size;

	public:

		layout() = default;

		explicit layout(const extents& shape)
		{
			assert(shape.size() == dims);
			std::copy(shape.begin(), shape.end(), _shape.begin());
			compute_strides();
		}

		template<class ... args_type, class = std::enable_if_t<dims == sizeof...(args_type)>>
		explicit layout(args_type&& ... dimension) :
			layout(extents(std::initializer_list<uint>{ (uint)dimension... }))
		{}

		constexpr uint shape(uint i) const { return _shape[i]; }
		constexpr uint stride(uint i) const { return _strides[i]; }

		extents shape() const { return extents(_shape); }
		extents strides() const { return extents(_strides); }
		uint size() const { return _size; }

		operator extents() const { return shape(); }

	private:

		void compute_strides()
		{
			for (uint i = 0; i < dims; i++)
			{
				uint stride = 1;
				for (uint j = dims - 1; j > i; j--)
					stride *= _shape[j];
				_strides[i] = stride;
			}
			_size = _strides[0] * _shape[0];
		}
	};

	template<uint dims>
	class tensor_slice
	{
		static_assert(dims > 0, "tensor dimensionality must be at least 1");

		scalar* _ptr;
		const uint* _shape;
		const uint* _strides;

		template<uint m>
		friend class tensor_slice;

	protected:

		inline tensor_slice(scalar* ptr, const uint* shape, const uint* strides) :
			_ptr(ptr), _shape(shape), _strides(strides)
		{}

	public:

		inline tensor_slice(const buffer& buf, const layout<dims>& layout) :
			_ptr(buf.ptr()),
			_shape(layout.shape().begin()),
			_strides(layout.strides().begin())
		{
			assert(buf.size() == layout.size());
		}

		tensor_slice(tensor_slice&& rhs) : tensor_slice((const tensor_slice&)rhs) {}
		tensor_slice(const tensor_slice&) = default;

		inline constexpr uint shape(uint i) const { return _shape[i]; }
		inline extents shape() const { return extents(_shape, _shape + dims); }
		scalar* ptr() const { return _ptr; }

		template<class ... args_type, class = std::enable_if_t<dims == sizeof...(args_type)>>
		scalar& at(args_type ... index) const
		{
			std::initializer_list<uint> indices_list = { (uint)index... };
			auto indices = indices_list.begin();
			auto ptr = _ptr;
			for (uint i = 0; i < dims; i++)
			{
				assert(indices[i] < _shape[i]);
				ptr += indices[i] * _strides[i];
			}
			return *ptr;
		}

		inline std::conditional_t<dims == 1, scalar&, tensor_slice<dims - 1>> operator[](uint i) const
		{
			assert(i < _shape[0]);
			if constexpr (dims == 1)
			{
				return _ptr[i];
			}
			else
			{
				return tensor_slice<dims - 1>(
					_ptr + (i * _strides[0]),
					_shape + 1,
					_strides + 1
				);
			}
		}

		friend class vector_slice;
	};

	namespace internal
	{
		template<uint dims>
		struct tensor_details
		{
			layout<dims> _layout;
			buffer       _data;

			tensor_details(const extents& shape) :
				_layout(shape),
				_data(_layout.size())
			{}
		};
	}

	template<uint dims>
	class tensor : private internal::tensor_details<dims>, public tensor_slice<dims>
	{
	public:

		tensor(const extents& shape) :
			tensor::tensor_details(shape),
			tensor::tensor_slice(this->_data, this->_layout)
		{}

		tensor(const layout<dims>& l) :
			tensor(l.shape())
		{}

		template<class ... args_type, class = std::enable_if_t<dims == sizeof...(args_type)>>
		tensor(args_type ... shape) :
			tensor(extents(std::initializer_list<uint>{ (uint)shape... }))
		{}

		tensor(tensor&&) = delete;
		tensor(const tensor&) = delete;

		const layout<dims>& layout() const { return this->_layout; }
		const buffer& data() const { return this->_data; }
		buffer& data() { return this->_data; }
	};


	class vector_slice : public tensor_slice<1>
	{
		static const uint stride = 1;

		uint _size;

	public:

		explicit vector_slice(const buffer& buf) :
			tensor_slice(buf.ptr(), &_size, &stride),
			_size((uint)buf.size())
		{}

		template<uint n>
		vector_slice(const tensor_slice<n>& slice) :
			tensor_slice(slice._ptr, &_size, &stride),
			_size(slice._shape[0] * slice._strides[0])
		{}

		uint size() const { return _size; }
	};

	inline vector_slice buffer::as_vector() const
	{
		return vector_slice(*this);
	}

	template<uint dims>
	inline tensor_slice<dims> buffer::as_tensor(const layout<dims>& l) const
	{
		return tensor_slice<dims>(*this, l);
	}

	/*************************************************************************************************************************************/

	template<typename function_type, typename ... args_type>
	using if_callable = std::enable_if_t<std::is_invocable_v<function_type, args_type...>, function_type>;

	class counting_iterator
	{
	private:

		uint count;

	public:

		using iterator_category = std::random_access_iterator_tag;
		using value_type = uint;
		using difference_type = int;
		using pointer = uint;
		using reference = uint;

		counting_iterator(uint c = 0) : count(c) {}
		counting_iterator(const counting_iterator& other) : count(other.count) {}

		//Arithmetic operations
		counting_iterator& operator++() { count++; return *this; }
		counting_iterator operator++(int) { counting_iterator tmp(*this); operator++(); return tmp; }

		uint operator-(counting_iterator s) const { return (count - s.count); }
		uint operator+(counting_iterator s) const { return (count + s.count); }
		counting_iterator& operator+=(uint s) { count += s; return *this; }
		counting_iterator& operator-=(uint s) { count -= s; return *this; }

		//Relational operations
		bool operator==(const counting_iterator& rhs) const { return count == rhs.count; }
		bool operator!=(const counting_iterator& rhs) const { return count != rhs.count; }

		//Accessor
		uint operator*() { return count; }
	};

	template<class K, class = if_callable<K, uint>>
	inline void dispatch(uint count, const K& kernel)
	{
		std::for_each(std::execution::par, counting_iterator(0), counting_iterator(count), kernel);
	}

	template<class K, class = if_callable<K, uint>>
	inline void dispatch(const layout<1>& layout, const K& kernel)
	{
		dispatch(layout.size(), kernel);
	}

	template<class K, class = if_callable<K, uint, uint>>
	inline void dispatch(const layout<2>& layout, const K& kernel)
	{
		dispatch(layout.size(), [&](uint i) {
			extents s = layout.shape();
			uint a = (i / layout.stride(0)) % layout.shape(0);
			uint b = (i / layout.stride(1)) % layout.shape(1);
			kernel(a, b);
		});
	}

	template<class K, class = if_callable<K, uint, uint, uint>>
	inline void dispatch(const layout<3>& layout, const K& kernel)
	{
		dispatch(layout.size(), [&](uint i) {
			extents s = layout.shape();
			uint a = (i / layout.stride(0)) % layout.shape(0);
			uint b = (i / layout.stride(1)) % layout.shape(1);
			uint c = (i / layout.stride(2)) % layout.shape(2);
			kernel(a, b, c);
		});
	}

	inline void tensor_fill(const vector_slice& x, scalar value)
	{
		dispatch(x.size(), [&](uint i)
		{
			x[i] = value;
		});
	}

	template<class F, class = std::enable_if_t<std::is_invocable_r_v<scalar, F>>>
	inline void tensor_fill(const vector_slice& x, const F& f)
	{
		dispatch(x.size(), [&](uint i)
		{
			x[i] = f();
		});
	}

	inline void tensor_zero(const vector_slice& x) { tensor_fill(x, 0.0f); }

	inline void tensor_update(const vector_slice& slice, const std::vector<scalar>& data)
	{
		assert(data.size() == slice.size());
		std::copy(data.begin(), data.end(), slice.ptr());
	}

	/*************************************************************************************************************************************/
}
