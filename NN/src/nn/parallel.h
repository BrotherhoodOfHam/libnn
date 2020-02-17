#pragma once

#include <execution>
#include <iterator>
#include <type_traits>

class counting_iterator
{
private:

	size_t count;

public:

	using iterator_category = std::random_access_iterator_tag;
	using value_type = size_t;
	using difference_type = std::make_signed_t<size_t>;
	using pointer = size_t;
	using reference = size_t;

	counting_iterator(size_t c = 0) : count(c) {}
	counting_iterator(const counting_iterator& other) : count(other.count) {}

	//Arithmetic operations
	counting_iterator& operator++() { count++; return *this; }
	counting_iterator operator++(int) { counting_iterator tmp(*this); operator++(); return tmp; }

	size_t operator-(counting_iterator s) const { return (count - s.count); }
	size_t operator+(counting_iterator s) const { return (count + s.count); }
	counting_iterator& operator+=(size_t s) { count += s; return *this; }
	counting_iterator& operator-=(size_t s) { count -= s; return *this; }

	//Relational operations
	bool operator==(const counting_iterator& rhs) const { return count == rhs.count; }
	bool operator!=(const counting_iterator& rhs) const { return count != rhs.count; }

	//Accessor
	size_t operator*() { return count; }
};

template<class F>
inline void parallel_for(size_t size, const F& callback)
{
	std::for_each(std::execution::seq, counting_iterator(0), counting_iterator(size), callback);
}

/*
int test()
{
	const size_t c = 10000;
	const size_t r = 10000;

	vector x(c);
	matrix m(r, c);
	vector y(r);

	auto time = std::chrono::high_resolution_clock::now();

	parallel_for(r, [&](size_t i_row) {
		for (size_t i_col = 0; i_col < c; i_col++)
		{
			y[i_row] = x[i_col] * m[i_row][i_col];
		}
		});

	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - time).count() << std::endl;

	return 0;
}
*/