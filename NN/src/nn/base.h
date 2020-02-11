/*
	Maths
*/

#pragma once

#include <vector>
#include <memory>
#include <cassert>

/*************************************************************************************************************************************/

typedef float scalar;

class vector
{
	std::vector<scalar> p;

public:

	size_t length;

	vector(size_t _length) :
		p(_length),
		length(_length)
	{}

	inline scalar operator[](size_t i) const { return p.at(i); }
	inline scalar& operator[](size_t i) { return p.at(i); }
};

class matrix
{
	std::vector<scalar> m;

public:

	size_t rows;
	size_t cols;

	template<typename T>
	class slice
	{
		T* p;
		size_t sz;

	public:

		slice(T* _p, size_t _sz) : p(_p), sz(_sz) {}

		inline T& operator[](size_t i)
		{
			assert(i < sz);
			return p[i];
		}

		inline T operator[](size_t i) const
		{
			assert(i < sz);
			return p[i];
		}
	};

	matrix(size_t _rows, size_t _cols) :
		m(_rows* _cols),
		rows(_rows),
		cols(_cols)
	{}

	matrix(matrix&& rhs) noexcept :
		m(std::move(rhs.m)),
		rows(rhs.rows),
		cols(rhs.cols)
	{}

	matrix(const matrix&) = delete;

	inline slice<scalar> operator[](size_t i)
	{
		assert(i < rows);
		return slice<scalar>(&m[cols * i], cols);
	}

	inline slice<const scalar> operator[](size_t i) const
	{
		assert(i < rows);
		return slice<const scalar>(&m[cols * i], cols);
	}
};

/*************************************************************************************************************************************/
