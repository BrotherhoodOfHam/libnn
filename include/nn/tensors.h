/*
	Tensor classes.
*/

#pragma once

#include <array>
#include <numeric>
#include <algorithm>
#ifdef __cpp_lib_parallel_algorithm
#include <execution>
#endif
#include <functional>

#include "common.h"

namespace nn
{
	using scalar = float;

	/*************************************************************************************************************************************/

	class tensor_shape : public const_span<uint>
	{
	public:

		using const_span<uint>::const_span;

		explicit tensor_shape(const uint& shape) : tensor_shape(&shape, &shape + 1) {}

		operator std::vector<uint>() { return std::vector<uint>(begin(), end()); }

		uint datasize() const
		{
			uint total = 1;
			for (uint i : *this)
				total *= i;
			return total;
		}

		static bool equals(const tensor_shape& lhs, const tensor_shape& rhs)
		{
			return std::equal(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
		}
	};

	using dynamic_tensor_shape = std::vector<uint>;

	/*************************************************************************************************************************************/

	class tensor_layout_error : std::runtime_error
	{
	public:

		tensor_layout_error(uint expected_dims, uint dims);
	};

	template<uint dims>
	class tensor_layout;

	template<uint dims>
	class tensor_layout_view
	{
		const uint* _shape = nullptr;
		const uint* _strides = nullptr;

		tensor_layout_view(const uint* shape, const uint* strides) :
			_shape(shape), _strides(strides)
		{}

	public:

		template<uint d>
		friend class tensor_layout_view;

		tensor_layout_view() = default;
		tensor_layout_view(const tensor_layout_view&) = default;
		tensor_layout_view(const tensor_layout<dims>& l);

		tensor_layout_view<dims - 1> sublayout() const
		{
			return tensor_layout_view<dims - 1>(_shape + 1, _strides + 1);
		}

		constexpr uint shape(uint i) const { return _shape[i]; }
		constexpr uint stride(uint i) const { return _strides[i]; }

		tensor_shape shape() const { return tensor_shape(_shape, _shape + dims); }
		tensor_shape strides() const { return tensor_shape(_strides, _strides + dims); }
		uint datasize() const { return _strides[0] * _shape[0]; }

		uint operator[](uint i) const { return shape(i); }
	};

	template<uint dims>
	class tensor_layout
	{
		std::array<uint, dims> _shape;
		std::array<uint, dims> _strides;

		void compute_strides()
		{
			// compute strides
			for (uint i = 0; i < dims; i++)
			{
				uint stride = 1;
				for (uint j = dims - 1; j > i; j--)
					stride *= _shape[j];
				_strides[i] = stride;
			}
		}

	public:

		tensor_layout() = default;
		tensor_layout(const tensor_layout&) = default;

		tensor_layout(const tensor_layout_view<dims>& view)
		{
			std::copy(view.shape().begin(), view.shape().end(), _shape.begin());
			std::copy(view.strides().begin(), view.strides().end(), _strides.begin());
		}

		tensor_layout(uint dim, const tensor_layout<dims - 1>& view)
		{
			_shape[0] = dim;
			std::copy(view.shape().begin(), view.shape().end(), _shape.begin() + 1);
			compute_strides();
		}

		tensor_layout(uint dim, tensor_shape shape) :
			tensor_layout(dim, tensor_layout<dims - 1>(shape))
		{}

		explicit tensor_layout(tensor_shape shape)
		{
			if (shape.size() != dims)
				throw tensor_layout_error(dims, (uint)shape.size());

			std::copy(shape.begin(), shape.end(), _shape.begin());
			compute_strides();
		}

		template<class ... args_type, class = std::enable_if_t<(dims-1) == sizeof...(args_type)>>
		explicit tensor_layout(uint dim, args_type ... dimension) :
			tensor_layout(tensor_shape(std::initializer_list<uint>{ dim, (uint)dimension... }))
		{}

		tensor_layout_view<dims - 1> sublayout() const { return tensor_layout_view<dims>(*this).sublayout(); }

		constexpr uint shape(uint i) const { return _shape[i]; }
		constexpr uint stride(uint i) const { return _strides[i]; }

		tensor_shape shape() const { return tensor_shape(_shape); }
		tensor_shape strides() const { return tensor_shape(_strides); }
		uint datasize() const { return _strides[0] * _shape[0]; }

		uint operator[](uint i) const { return shape(i); }
	};

	template<uint dims>
	tensor_layout_view<dims>::tensor_layout_view(const tensor_layout<dims>& l) :
		_shape(l.shape().begin()),
		_strides(l.strides().begin())
	{}

	/*************************************************************************************************************************************/

	enum class tensor_kind
	{
		is_object, // owns it's layout
		is_slice  // holds a reference to an existing layout
	};

	template<uint rank, typename element_type = scalar, tensor_kind kind = tensor_kind::is_object>
	class tensor;

	using vector = tensor<1>;

	/*
		Tensor view class.
		Represents an multidimensional array view on some memory
	*/
	template<uint rank, typename _element_type, tensor_kind kind>
	class tensor
	{
	public:

		static constexpr bool is_vector = rank == 1;

		using layout_type = std::conditional_t<kind == tensor_kind::is_slice, tensor_layout_view<rank>, tensor_layout<rank>>;
		using element_type = _element_type;

		template<uint _rank>
		using view_type = tensor<_rank, element_type, tensor_kind::is_object>;

		using slice = tensor<rank - 1, scalar, tensor_kind::is_slice>;
		using const_slice = tensor<rank - 1, const scalar, tensor_kind::is_slice>;

		//static_assert(rank > 0, "tensor rank must be 1 or higher");

		tensor() = default;
		tensor(element_type* ptr, const tensor_layout_view<rank>& layout) :
			_ptr(ptr), _layout(layout)
		{}

		template<tensor_kind _kind>
		tensor(const tensor<rank, element_type, _kind>& rhs) :
			_ptr(rhs.ptr()),
			_layout(rhs.layout())
		{}

		template<uint _rank, tensor_kind _kind, bool _isvec = is_vector, typename = std::enable_if_t<_isvec>>
		tensor(const tensor<_rank, element_type, _kind>& rhs) :
			_ptr(rhs.ptr()),
			_layout(rhs.size())
		{}

		template<uint dims>
		view_type<dims> reshape(const tensor_layout_view<dims>& ly) const
		{
			return reshape(tensor_layout<dims>(ly));
		}

		template<uint dims>
		view_type<dims> reshape(const tensor_layout<dims>& ly) const
		{
			assert(ly.datasize() == size());
			return view_type<dims>(_ptr, ly);
		}

		template<typename ... args_t, uint dims = sizeof...(args_t)+1>
		view_type<dims> reshape(uint shape0, args_t ... shape) const
		{
			tensor_layout<dims> ly(shape0, shape...);
			assert(ly.datasize() == datasize());
			return view_type<dims>(_ptr, ly);
		}

		vector flatten() const { return vector(_ptr, nn::tensor_layout<1>(_layout.datasize())); }
		
		inline scalar* ptr() const { return _ptr; }
		inline uint shape(uint i) const { return _layout.shape(i); }
		inline uint stride(uint i) const { return _layout.stride(i); }
		inline tensor_shape shape() const { return _layout.shape(); }
		inline tensor_layout<rank> layout() const { return _layout; }
		inline uint size() const { return _layout.datasize(); }

		template<uint _rank = rank, typename = std::enable_if_t<(_rank > 1)>>
		inline const_slice operator[](uint index) const
		{
			assert(index < shape(0));
			return const_slice(
				_ptr + ((size_t)index * stride(0)),
				_layout.sublayout()
			);
		}
		
		template<uint _rank = rank, typename = std::enable_if_t<(_rank > 1)>>
		inline slice operator[](uint index)
		{
			assert(index < shape(0));
			return slice(
				_ptr + ((size_t)index * stride(0)),
				_layout.sublayout()
			);
		}

		template<uint _rank = rank, typename = std::enable_if_t<(_rank == 1)>>
		inline const element_type& operator[](uint index) const
		{
			assert(index < shape(0));
			return _ptr[index];
		}

		template<uint _rank = rank, typename = std::enable_if_t<(_rank == 1)>>
		inline element_type& operator[](uint index)
		{
			assert(index < shape(0));
			return _ptr[index];
		}

	private:

		element_type* _ptr = nullptr;
		layout_type _layout;
	};
	
	/*************************************************************************************************************************************/

	/*
#ifdef __cpp_lib_parallel_algorithm
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

	inline void tensor_fill(const slice& x, scalar value)
	{
		dispatch(x.size(), [&](uint i)
		{
			x[i] = value;
		});
	}

	template<class F, class = std::enable_if_t<std::is_invocable_r_v<scalar, F>>>
	inline void tensor_fill(const slice& x, const F& f)
	{
		dispatch(x.size(), [&](uint i)
		{
			x[i] = f();
		});
	}

	inline void tensor_zero(const slice& x) { tensor_fill(x, 0.0f); }

	template<typename container_type>
	inline void tensor_update(const slice& slice, const container_type& data)
	{
		assert(data.size() == slice.size());
		std::copy(data.begin(), data.end(), slice.ptr());
	}
#endif
	*/

	/*************************************************************************************************************************************/
}
