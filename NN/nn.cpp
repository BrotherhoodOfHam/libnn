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
		m(_rows * _cols),
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


class nn_node
{
private:

	size_t _input_size;
	size_t _output_size;

public:

	nn_node(size_t input_size, size_t output_size) :
		_input_size(input_size), _output_size(output_size)
	{}

	nn_node(const nn_node&) = delete;

	size_t input_size() const { return _input_size; }
	size_t output_size() const { return _output_size; }

	virtual void forward(const vector& x, vector& y) const = 0;

	virtual void backward(const vector& y, const vector& x, const vector& dy, vector& dx, matrix& dw, vector& db) const = 0;

	virtual void update_params(const matrix& dw, const vector& db, float r, float k) = 0;
};

class nn_activation : public nn_node
{
	activation _type;
	bool _isoutput;

public:

	nn_activation(size_t input_size, activation type, bool is_output) :
		_type(type), _isoutput(is_output),
		nn_node(input_size, input_size)
	{}

	void forward(const vector& x, vector& y) const override
	{
		assert(x.length == y.length);

		switch (_type)
		{
			case activation::linear:
			{
				for (size_t i = 0; i < x.length; i++)
					y[i] = x[i];
				break;
			}
			case activation::relu:
			{
				for (size_t i = 0; i < x.length; i++)
					y[i] = std::max(x[i], 0.0f);
				break;
			}
			case activation::sigmoid:
			{
				for (size_t i = 0; i < x.length; i++)
					y[i] = 1.0f / (1.0f + std::exp(-x[i]));
				break;
			}
			case activation::softmax:
			{
				scalar sum = 0.0f;
				for (size_t i = 0; i < x.length; i++)
				{
					y[i] = std::exp(x[i]);
					sum += y[i];
				}
				for (size_t i = 0; i < x.length; i++)
					y[i] /= sum;
				break;
			}
		}
	}

	void backward(const vector& y, const vector& x, const vector& dy, vector& dx, matrix& dw, vector& db) const override
	{
		assert(x.length == y.length);
		assert(x.length == dy.length);
		assert(x.length == dx.length);

		//workaround for when this is the last layer
		if (_isoutput)
		{
			for (size_t i = 0; i < x.length; i++)
				dx[i] = dy[i];
			return;
		}

		switch (_type)
		{
			case activation::linear:
			{
				for (size_t i = 0; i < x.length; i++)
					dx[i] = dy[i];
				break;
			}
			case activation::relu:
			{
				for (size_t i = 0; i < x.length; i++)
					if (x[i] > 0.0f)
						dx[i] = dy[i];
					else
						dx[i] = 0.0f;
				break;
			}
			case activation::sigmoid:
			{
				for (size_t i = 0; i < x.length; i++)
				{
					dx[i] = (y[i] * (1.0f - y[i])) * dy[i];
				}
				break;
			}
			case activation::softmax:
			{
				for (size_t j = 0; j < x.length; j++)
				{
					float sum = 0.0f;
					for (size_t i = 0; i < x.length; i++)
					{
						float dydz = (i == j)
							? y[i] * (1 - y[i])
							: -y[i] * y[j];

						sum += dy[i] * dydz;
					}
					dx[j] = sum;
				}
				break;
			}
		}
	}

	void update_params(const matrix& dw, const vector& db, float k, float r) override { }
};

class nn_layer : public nn_node
{
	matrix w;
	vector b;

public:

	nn_layer(size_t input_size, size_t layer_size) :
		w(layer_size, input_size),
		b(layer_size),
		nn_node(input_size, layer_size)
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

	void forward(const vector& x, vector& y) const override
	{
		assert(y.length == w.rows);
		assert(x.length == w.cols);

		//for each row:
		//y = w.x + b
		for (size_t j = 0; j < w.rows; j++)
		{
			//z = w.x + b
			scalar z = 0.0f;
			for (size_t i = 0; i < w.cols; i++)
				z += x[i] * w[j][i];
			z += b[j];
			y[j] = z;
		}
	}

	void backward(const vector& y, const vector& x, const vector& dy, vector& dx, matrix& dw, vector& db) const override
	{
		assert(dw.rows == w.rows && dw.cols == w.cols);
		assert(db.length == b.length);
		assert(dx.length == input_size());
		assert(dy.length == output_size());
		assert(x.length == input_size());
		assert(y.length == output_size());

		// δ/δy = partial derivative of loss w.r.t to output
		// the goal is to find the derivatives w.r.t to parameters: δw, δb, δx

		// compute partial derivatives w.r.t weights and biases
		for (size_t j = 0; j < w.rows; j++)
		{
			// δb = δy
			db[j] = dy[j];
			// δw = δy * x 
			for (size_t i = 0; i < x.length; i++)
				dw[j][i] = x[i] * dy[j];
		}

		// compute partial derivative w.r.t input x
		for (size_t i = 0; i < w.cols; i++)
		{
			// δx = w^T * δy
			scalar s = 0.0f;
			for (size_t j = 0; j < w.rows; j++)
				s += w[j][i] * dy[j];
			dx[i] = s;
		}
	}

	void update_params(const matrix& dw, const vector& db, float k, float r) override
	{
		assert(dw.rows == w.rows && dw.cols == w.cols);
		assert(db.length == b.length);

		// apply gradient descent
		for (size_t j = 0; j < w.rows; j++)
		{
			b[j] -= k * db[j];

			for (size_t i = 0; i < w.cols; i++)
				w[j][i] = (r * w[j][i]) - (k * dw[j][i]);
		}
	}
};

/*************************************************************************************************************************************/

class nn
{
	std::vector<std::unique_ptr<nn_node>> _layers;
	std::vector<vector> _a;  // layer activations
	std::vector<vector> _dy; // layer gradient
	std::vector<matrix> _dw; // layer weight gradient
	std::vector<vector> _db; // layer bias gradient

public:

	nn(size_t input_size, std::vector<layer> layers)
	{
		_layers.reserve(layers.size() * 2);
		_a.reserve(_layers.size() + 1);
		_dy.reserve(_layers.size() + 1);
		_dw.reserve(_layers.size());
		_db.reserve(_layers.size());

		_a.emplace_back(input_size);
		_dy.emplace_back(input_size);

		size_t i = 0;
		size_t layer_input = input_size;

		for (const auto& layer_info : layers)
		{
			i++;
			
			_layers.push_back(std::make_unique<nn_layer>(layer_input, layer_info.size));
			_a.emplace_back(layer_info.size);
			_dy.emplace_back(layer_info.size);
			_dw.emplace_back(layer_info.size, layer_input);
			_db.emplace_back(layer_info.size);

			_layers.push_back(std::make_unique<nn_activation>(layer_info.size, layer_info.actv, i == layers.size()));
			_a.emplace_back(layer_info.size);
			_dy.emplace_back(layer_info.size);
			_dw.emplace_back(0, 0);
			_db.emplace_back(0);

			layer_input = layer_info.size;
		}
	}

	nn(const nn&) = delete;

	void train(std::vector<std::pair<vector, vector>> training_set, std::vector<std::pair<vector, vector>> testing_set, size_t epochs, float learning_rate, float regularization_term)
	{
		const float k = learning_rate;
		const float r = 1 - (learning_rate * regularization_term / training_set.size());

		for (size_t ep = 0; ep < epochs; ep++)
		{
			std::cout << "epoch " << ep << ":" << std::endl;

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

				train_batch(p.first, p.second, k, r);

				c++;
			}

			double error = 0.0;
			size_t correct = 0;
			for (const auto& p : testing_set)
			{
				const auto& prediction = forward(p.first);

				for (size_t i = 0; i < prediction.length; i++)
					error += std::pow(prediction[i] - p.second[i], 2);
				
				if (arg_max(prediction) == arg_max(p.second))
					correct++;
			}
			error /= (2.0 * testing_set.size());

			std::cout << "mse: " << error << " | recognised: (" << correct << " / " << testing_set.size() << ")" << std::endl;
		}
	}

	void train_batch(const vector& x, const vector& y, float k, float r)
	{
		assert(_layers.back()->output_size() == y.length);
		assert(_layers.front()->input_size() == x.length);

		//forward prop
		for (size_t i = 0; i < x.length; i++)
			_a[0][i] = x[i];

		for (size_t i = 0; i < _layers.size(); i++)
			_layers[i]->forward(_a[i], _a[i + 1]);

		//backward prop
		loss_derivative(_a.back(), y, _dy.back());

		for (int i = _layers.size() - 1; i >= 0; i--)
			_layers[i]->backward(_a[i + 1], _a[i], _dy[i + 1], _dy[i], _dw[i], _db[i]);

		//gradient descent optimization
		for (size_t i = 0; i < _layers.size(); i++)
			_layers[i]->update_params(_dw[i], _db[i], k, r);
	}

	const vector& forward(const vector& x)
	{
		for (size_t i = 0; i < x.length; i++)
			_a[0][i] = x[i];

		for (size_t i = 0; i < _layers.size(); i++)
			_layers[i]->forward(_a[i], _a[i + 1]);

		return _a.back();
	}

private:

	void loss_derivative(const vector& y, const vector& t, vector& dy)
	{
		assert(dy.length == t.length);
		assert(dy.length == y.length);

		for (size_t i = 0; i < dy.length; i++)
		{
			dy[i] = y[i] - t[i];
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