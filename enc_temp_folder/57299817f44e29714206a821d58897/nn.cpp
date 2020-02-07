#include <cmath>
#include <cassert>
#include <memory>
#include <iostream>
#include <random>
#include <chrono>

#include "mnist/mnist_reader_less.hpp"

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

enum class activation
{
	linear,
	sigmoid,
	relu,
	softmax
};

struct layer
{
	size_t size;
	activation actv;

	layer(size_t _size, activation _actv = activation::sigmoid) :
		size(_size),
		actv(_actv)
	{}
};

class nn_layer
{
	matrix w;
	vector b;
	activation activation_type;
	bool is_output;

public:

	struct actvn
	{
		vector a;
		vector z;

		actvn(const actvn&) = delete;

		actvn(vector _a, vector _z) :
			a(std::move(_a)), z(std::move(_z))
		{}

		actvn(actvn&& rhs) noexcept :
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

	nn_layer(size_t layer_size, size_t input_size, activation _activation_type, bool _is_output) :
		w(layer_size, input_size),
		b(layer_size),
		activation_type(_activation_type),
		is_output(_is_output)
	{
		std::default_random_engine gen;
		std::normal_distribution<float> dist(0, 1);
		const float sqrtn = std::sqrt((float)input_size);

		for (size_t j = 0; j < layer_size; j++)
			for (size_t i = 0; i < input_size; i++)
				w[j][i] = dist(gen) / sqrtn;

		for (size_t i = 0; i < layer_size; i++)
			b[i] = 0.0f;
	}

	inline size_t input_size() const { return w.cols; }
	inline size_t layer_size() const { return w.rows; }

	actvn alloc_activation() const { return actvn(vector(layer_size()), vector(layer_size())); }
	delta alloc_delta() const { return delta(matrix(w.rows, w.cols), vector(b.length), vector(input_size())); }

	void forward(actvn& out, const vector& x) const
	{
		assert(out.a.length == w.rows);
		assert(out.z.length == w.rows);
		assert(x.length == w.cols);

		//for each activation a:
		//a = f(w.x + b)
		for (size_t j = 0; j < w.rows; j++)
		{
			//z = w.x + b
			scalar z = 0.0f;
			for (size_t i = 0; i < w.cols; i++)
				z += x[i] * w[j][i];
			z += b[j];

			//cache z and a
			out.z[j] = z;
		}

		activate(out.z, out.a);
	}

	void backward(delta& out, const vector& dy, const vector& x, const vector& z, const vector& y) const
	{
		assert(out.dw.rows == w.rows && out.dw.cols == w.cols);
		assert(out.db.length == b.length);
		assert(out.dx.length == input_size());
		assert(dy.length == layer_size());
		assert(x.length == input_size());
		assert(z.length == layer_size());

		// δ/δy = partial derivative of cost w.r.t to output activations (δC/δy)
		// the goal is to find the derivatives w.r.t to parameters: δw, δb, δx

		if (is_output)
		{
			// in the case of the output layer we will assume: δb = δy
			for (size_t i = 0; i < dy.length; i++)
				out.db[i] = dy[i];
		}
		else
		{
			// store δz in δb
			derivative(z, y, dy, out.db);
		}
		
		// compute partial derivatives w.r.t weights and biases
		for (size_t j = 0; j < w.rows; j++)
		{
			// δw = δz * x 
			for (size_t i = 0; i < x.length; i++)
				out.dw[j][i] = out.db[j] * x[i];
		}

		// compute partial derivative w.r.t input x
		for (size_t i = 0; i < w.cols; i++)
		{
			// δx = w^T * δz 
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

		//apply gradient descent
		for (size_t j = 0; j < w.rows; j++)
		{
			b[j] -= k * d.db[j];

			for (size_t i = 0; i < w.cols; i++)
				w[j][i] = (r * w[j][i]) - (k * d.dw[j][i]);
		}
	}

private:

	// compute activation
	// a = f(z)
	void activate(const vector& z, vector& a) const
	{
		assert(z.length == a.length);

		switch (activation_type)
		{
			case activation::linear:
			{
				for (size_t i = 0; i < z.length; i++)
					a[i] = z[i];
				break;
			}
			case activation::relu:
			{
				for (size_t i = 0; i < z.length; i++)
					a[i] = std::max(z[i], 0.0f);
				break;
			}
			case activation::sigmoid:
			{
				for (size_t i = 0; i < z.length; i++)
					a[i] = 1.0f / (1.0f + std::exp(-z[i]));
				break;
			}
			case activation::softmax:
			{
				scalar sum = 0.0f;
				for (size_t i = 0; i < z.length; i++)
				{
					a[i] = std::exp(z[i]);
					sum += a[i];
				}
				for (size_t i = 0; i < z.length; i++)
					a[i] /= sum;
				break;
			}
		}
	}

	// compute partial derivative w.r.t to z
	// δ/δz = δy/δz * δ/δy
	void derivative(const vector& z, const vector& y, const vector& dy, vector& dz) const
	{
		assert(z.length == y.length);
		assert(z.length == dy.length);
		assert(z.length == dz.length);

		switch (activation_type)
		{
			case activation::linear:
			{
				for (size_t i = 0; i < z.length; i++)
					dz[i] = dy[i];
				break;
			}
			case activation::relu:
			{
				for (size_t i = 0; i < z.length; i++)
					if (z[i] > 0.0f)
						dz[i] = dy[i];
					else
						dz[i] = 0.0f;
				break;
			}
			case activation::sigmoid:
			{
				for (size_t i = 0; i < z.length; i++)
				{
					dz[i] = (y[i] * (1.0f - y[i])) * dy[i];
				}
				break;
			}
			case activation::softmax:
			{
				for (size_t j = 0; j < z.length; j++)
				{
					float sum = 0.0f;
					for (size_t i = 0; i < z.length; i++)
					{
						float dydz = (i == j)
							? y[i] * (1 - y[i])
							: -y[i] * y[j];

						sum += dy[i] * dydz;
					}
					dz[j] = sum;
				}
				break;
			}
		}
	}
};

class nn
{
	std::vector<nn_layer> _layers;
	std::vector<nn_layer::delta> _deltas;
	std::vector<nn_layer::actvn> _actvns;

public:

	nn(size_t input_size, std::vector<layer> layers)
	{
		_layers.reserve(layers.size());
		_deltas.reserve(layers.size());
		_actvns.reserve(layers.size());

		_actvns.push_back(nn_layer::actvn(vector(input_size), vector(1)));

		size_t i = 0;
		size_t layer_input = input_size;

		for (const auto& layer : layers)
		{
			i++;
			_layers.emplace_back(layer.size, layer_input, layer.actv, layers.size() == i);
			_actvns.push_back(_layers.back().alloc_activation());
			_deltas.push_back(_layers.back().alloc_delta());

			layer_input = layer.size;
		}
	}

	nn(const nn&) = delete;

	void train(std::vector<std::pair<vector, vector>> training_set, std::vector<std::pair<vector, vector>> testing_set, size_t epochs, float learning_rate, float regularization_term)
	{
		const float k = learning_rate;
		const float r = 1 - (learning_rate * regularization_term / training_set.size());

		for (size_t ep = 0; ep < epochs; ep++)
		{
			std::cout << "epoch " << ep << ":\n" << std::endl;

			std::shuffle(training_set.begin(), training_set.end(), std::default_random_engine(ep));

			auto first = std::chrono::system_clock::now();
			auto last = first;
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

			double mse = 0.0;
			size_t correct = 0;
			for (const auto& p : testing_set)
			{
				const auto& prediction = forward(p.first).back().a;

				float mse = 0.0f;
				for (size_t i = 0; i < prediction.length; i++)
					mse += std::pow(prediction[i] - p.second[i], 2);
				
				if (arg_max(prediction) == arg_max(p.second))
					correct++;
			}
			mse /= (2.0 * testing_set.size());

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

	const std::vector<nn_layer::actvn>& forward(const vector& x)
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

	const std::vector<nn_layer::delta>& backprop(const vector& x, const vector& y)
	{
		assert(_actvns.size() == _layers.size() + 1);
		assert(_deltas.size() == _layers.size());
		assert(_layers.back().layer_size() == y.length);
		assert(_layers.front().input_size() == x.length);

		const auto& actvns = forward(x);

		assert(actvns.front().a.length == x.length);
		assert(actvns.back().a.length == y.length);
		
		vector loss(_layers.back().layer_size());
		loss_derivative(actvns.back().a, y, loss);

		auto dy = std::ref(loss);

		for (int i = _layers.size() - 1; i >= 0; i--)
		{
			_layers[i].backward(_deltas[i], dy, actvns[i].a, actvns[i + 1].z, actvns[i + 1].a);
			dy = std::ref(_deltas[i].dx);
		}

		return _deltas;
	}

	void loss_derivative(const vector& y, const vector& t, vector& dy)
	{
		assert(dy.length == t.length);
		assert(dy.length == y.length);

		for (size_t i = 0; i < dy.length; i++)
		{
			dy[i] = y[i] - t[i];
		}
	}
};

std::vector<std::pair<vector, vector>> preprocess(const std::vector<std::vector<uint8_t>>& data, const std::vector<uint8_t>& labels)
{
	std::cout << "Preprocess" << std::endl;

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
	std::cout << "Load MNIST dataset" << std::endl;
	auto dataset = mnist::read_dataset<uint8_t, uint8_t>();

	nn n(784, {
		//layer(100, activation::relu),
		layer(32, activation::relu),
		layer(10, activation::softmax)
	});
	n.train(
		preprocess(dataset.training_images, dataset.training_labels),
		preprocess(dataset.test_images, dataset.test_labels),
		10,
		0.01f,
		5
	);

	return 0;
}

/*************************************************************************************************************************************/