#include <cmath>
#include <cassert>
#include <memory>
#include <iostream>
#include <random>
#include <chrono>

#include "mnist/mnist_reader_less.hpp"
#include <Windows.h>

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

class layer
{
	matrix w;
	vector b;
	bool is_output;

public:

	struct activation
	{
		vector a;
		vector z;

		activation(const activation&) = delete;

		activation(vector _a, vector _z) :
			a(std::move(_a)), z(std::move(_z))
		{}

		activation(activation&& rhs) noexcept :
			a(std::move(rhs.a)), z(std::move(rhs.z))
		{}
	};

	struct delta
	{
		matrix dw;
		vector db;
		vector dx;

		delta(const delta&) = delete;

		delta(matrix _dw, vector _db, vector _dx) :
			dw(std::move(_dw)), db(std::move(_db)), dx(std::move(_dx))
		{}

		delta(delta&& rhs) noexcept :
			dw(std::move(rhs.dw)), db(std::move(rhs.db)), dx(std::move(rhs.dx))
		{}
	};

	layer(size_t layer_size, size_t input_size, bool _is_output = false) :
		w(layer_size, input_size),
		b(layer_size),
		is_output(_is_output)
	{
		std::default_random_engine gen;
		std::normal_distribution<float> dist(0, 1);
		const float sqrtn = std::sqrt(input_size);

		for (size_t j = 0; j < layer_size; j++)
			for (size_t i = 0; i < input_size; i++)
				w[j][i] = dist(gen) / sqrtn;

		for (size_t i = 0; i < layer_size; i++)
			b[i] = 0.0f;
	}

	inline size_t input_size() const { return w.cols; }
	inline size_t layer_size() const { return w.rows; }

	activation alloc_activation() const
	{
		return activation(vector(layer_size()), vector(layer_size()));
	}

	delta alloc_delta() const
	{
		return delta(matrix(w.rows, w.cols), vector(b.length), vector(input_size()));
	}

	void forward(activation& out, const vector& x) const
	{
		assert(out.a.length == w.rows);
		assert(out.z.length == w.rows);
		assert(x.length == w.cols);

		//a = s(W.x + b)
		for (size_t j = 0; j < w.rows; j++)
		{
			scalar z = 0.0f;
			for (size_t i = 0; i < w.cols; i++)
				z += x[i] * w[j][i];
			z += b[j];

			out.z[j] = z;
			out.a[j] = sigmoid(z);
		}
	}

	void backward(delta& out, const vector& dy, const vector& x, const vector& z) const
	{
		assert(out.dw.rows == w.rows && out.dw.cols == w.cols);
		assert(out.db.length == b.length);
		assert(out.dx.length == input_size());
		assert(dy.length == layer_size());
		assert(x.length == input_size());
		assert(z.length == layer_size());

		//db
		for (size_t i = 0; i < b.length; i++)
			out.db[i] = dy[i] * sigmoidDerivative(z[i]);

		//dw - outer product
		for (size_t j = 0; j < out.db.length; j++)
			for (size_t i = 0; i < x.length; i++)
				out.dw[j][i] = out.db[j] * x[i];

		//dx
		for (size_t i = 0; i < w.cols; i++)
		{
			scalar s = 0.0f;
			for (size_t j = 0; j < w.rows; j++)
				s += out.db[j] * w[j][i];
			out.dx[i] = s;
		}
	}

	void update_params(const delta& d, float k, float r)
	{
		assert(d.dw.rows == w.rows && d.dw.cols == w.cols);
		assert(d.db.length == b.length);

		for (size_t i = 0; i < b.length; i++)
			b[i] -= k * d.db[i];

		for (size_t j = 0; j < w.rows; j++)
			for (size_t i = 0; i < w.cols; i++)
				w[j][i] = (r * w[j][i]) - (k * d.dw[j][i]);
	}

private:

	scalar sigmoid(scalar z) const
	{
		return 1.0f / (1.0f + std::exp(-z));
	}

	scalar sigmoidDerivative(scalar z) const
	{
		if (is_output)
			return 1.0f;
		scalar s = sigmoid(z);
		return s * (1.0f - s);
	}
};

class nn
{
	std::vector<layer> _layers;
	std::vector<layer::delta> _deltas;
	std::vector<layer::activation> _actvns;

public:

	nn(std::vector<size_t> layer_sizes)
	{
		_layers.reserve(layer_sizes.size());
		_deltas.reserve(layer_sizes.size());
		_actvns.reserve(layer_sizes.size());

		_actvns.push_back(layer::activation(vector(layer_sizes[0]), vector(1)));

		for (size_t i = 1; i < layer_sizes.size(); i++)
		{
			_layers.emplace_back(layer_sizes[i], layer_sizes[i - 1], (i + 1) == layer_sizes.size());
			_actvns.push_back(_layers.back().alloc_activation());
			_deltas.push_back(_layers.back().alloc_delta());
		}
	}

	nn(const nn&) = delete;

	void train(std::vector<std::pair<vector, vector>> training_set, std::vector<std::pair<vector, vector>> testing_set, size_t epochs, float learning_rate, float regularization_term)
	{
		const float k = learning_rate;
		const float r = 1 - (learning_rate * regularization_term / training_set.size());

		std::default_random_engine rng;

		for (uint32_t ep = 0; ep < epochs; ep++)
		{
			std::cout << "epoch " << ep << ":\n" << std::endl;
			std::cout << "training..." << std::endl;

			std::random_shuffle(testing_set.begin(), testing_set.end());

			auto last = std::chrono::system_clock::now();
			size_t c = 0;
			for (const auto& p : training_set)
			{
				auto t = std::chrono::system_clock::now();
				if ((t - last) > std::chrono::seconds(1))
				{
					std::cout << "(" << c << "/" << training_set.size() << ") ";
					std::cout << std::round(1000.0f * (float)c / training_set.size()) / 10 << "%" << std::endl;
					last = t;
				}

				const auto& deltas = backprop(p.first, p.second);
				for (size_t i = 0; i < _layers.size(); i++)
				{
					_layers[i].update_params(deltas[i], k, r);
				}

				c++;
			}

			std::cout << "testing..." << std::endl;

			std::random_shuffle(testing_set.begin(), testing_set.end());

			float mse = 0.0f;
			uint32_t correct = 0;
			for (const auto& p : testing_set)
			{
				const auto& prediction = forward(p.first).back().a;

				float mse = 0.0f;
				for (size_t i = 0; i < prediction.length; i++)
					mse += std::pow(prediction[i] - p.second[i], 2);
				
				if (arg_max(prediction) == arg_max(p.second))
					correct++;
			}
			mse /= (2 * testing_set.size());

			std::cout << "mse: " << mse << " | recognised: (" << correct << " / " << testing_set.size() << ")" << std::endl;
		}
	}

	size_t arg_max(const vector& v)
	{
		if (v.length == 0)
			return -1;

		float max = v[0];
		size_t i_max = 0;
		for (size_t i = 0; i < v.length; i++)
		{
			if (v[i] > max)
			{
				max = v[i];
				i_max = i;
			}
		}
		return i_max;
	}

	const std::vector<layer::activation>& forward(const vector& x)
	{
		assert(x.length == _layers.front().input_size());
		assert(x.length == _actvns[0].a.length);

		_actvns[0].a = x;

		for (size_t i = 0; i < _layers.size(); i++)
		{
			_layers[i].forward(_actvns[i+1], _actvns[i].a);
		}

		return _actvns;
	}

private:

	const std::vector<layer::delta>& backprop(const vector& x, const vector& y)
	{
		assert(_actvns.size() == _layers.size() + 1);
		assert(_deltas.size() == _layers.size());
		assert(_layers.back().layer_size() == y.length);
		assert(_layers.front().input_size() == x.length);

		const auto& actvns = forward(x);

		assert(actvns.front().a.length == x.length);
		assert(actvns.back().a.length == y.length);
		
		vector dcost(_layers.back().layer_size());
		for (size_t i = 0; i < dcost.length; i++)
			dcost[i] = actvns.back().a[i] - y[i];

		auto dy = std::ref(dcost);

		for (int i = _layers.size() - 1; i >= 0; i--)
		{
			_layers[i].backward(_deltas[i], dy, actvns[i].a, actvns[i + 1].z);
			dy = std::ref(_deltas[i].dx);
		}

		return _deltas;
	}
};

std::vector<std::pair<vector, vector>> preprocess(const std::vector<std::vector<uint8_t>>& data, const std::vector<uint8_t>& labels)
{
	std::cout << "preprocessing..." << std::endl;

	std::vector<std::pair<vector, vector>> t;
	t.reserve(data.size());

	for (size_t i = 0; i < data.size(); i++)
	{
		const auto& d = data[i];
		const uint8_t l = labels[i];

		vector dv(d.size());
		vector lv(10);

		for (size_t i = 0; i < d.size(); i++)
			dv[i] = (float)d[i] / 255.0f;

		lv[l] = 1.0f;

		t.push_back(std::make_pair(std::move(dv), std::move(lv)));
	}
	return std::move(t);
}

int main()
{ 
	std::cout << "Loading MNIST" << std::endl;
	SetCurrentDirectoryA("F:\\dev\\nn2");

	auto dataset = mnist::read_dataset<uint8_t, uint8_t>();

	nn n({ 784, 32, 10 });
	n.train(
		preprocess(dataset.training_images, dataset.training_labels),
		preprocess(dataset.test_images, dataset.test_labels),
		10,
		0.1f,
		5
	);

	return 0;
}

/*************************************************************************************************************************************/