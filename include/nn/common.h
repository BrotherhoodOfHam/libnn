/*
	Common headers and functions
*/

#pragma once

#include <vector>
#include <array>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <cassert>
#include <functional>

namespace nn
{
	using byte = unsigned char;
	using uint = unsigned int;

	/*
		Print current time
	*/
	std::ostream& time_stamp(std::ostream& out);

	/*
		Random engine helper
	*/
	inline std::default_random_engine new_random_engine()
	{
		return std::default_random_engine(std::random_device()());
	}

	/*
		Container span
	*/
	template<typename type>
	class span
	{
		type* _begin;
		type* _end;

	public:

		using value_type = type;
		using reference = type&;
		using const_reference = const type&; 
		using size_type = size_t;

		using iterator = type*;
		using const_iterator = const type*;

		span(type* begin, type* end) :
			_begin(begin), _end(end)
		{}

		span(const std::initializer_list<type>& values) :
			span(values.begin(), values.end())
		{}
		span(std::initializer_list<type>& values) :
			span(values.begin(), values.end())
		{}

		span(const std::vector<std::remove_const_t<type>>& values) :
			span(values.data(), values.data() + values.size())
		{}
		span(std::vector<type>& values) :
			span(values.data(), values.data() + values.size())
		{}

		template<size_t n>
		span(const std::array<std::remove_const_t<type>, n>& values) :
			span(values.data(), values.data() + n)
		{}
		template<size_t n>
		span(std::array<type, n>& values) :
			span(values.data(), values.data() + n)
		{}

		reference at(size_t i)
		{
			assert((_begin + i) < _end);
			return _begin[i];
		}

		const_reference at(size_t i) const
		{
			assert((_begin + i) < _end);
			return _begin[i];
		}

		size_type size() const { return _end - _begin; }

		iterator begin() { return _begin; }
		iterator end() { return _end; }

		const_iterator begin() const { return _begin; }
		const_iterator end() const { return _end; }

		reference operator[](size_type i) { return at(i); }
		const_reference operator[](size_type i) const { return at(i); }
	};

	template<typename type>
	using const_span = span<const type>;

	/*
		Type traits
	*/

	template <typename F, typename... Args>
	struct is_callable :
		std::is_constructible<
		std::function<void(Args ...)>,
		std::reference_wrapper<typename std::remove_reference<F>::type>
		>
	{};

	template<typename function_type, typename ... args_type>
	using if_callable = std::enable_if_t<is_callable<function_type, args_type...>::value, function_type>;

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
}
