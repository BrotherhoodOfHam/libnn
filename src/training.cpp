/*
	Neural Network Model
*/

#include <cmath>
#include <memory>
#include <iostream>
#include <random>
#include <chrono>

#include "device/gpu.h"
#include "nn/training.h"

using namespace nn;

/*************************************************************************************************************************************/

static size_t arg_max(const tensor<1>& v)
{
	float max = v[0];
	uint i_max = 0;
	for (uint i = 0; i < v.size(); i++)
	{
		if (v[i] > max)
		{
			max = v[i];
			i_max = i;
		}
	}
	return i_max;
}

static size_t arg_max(const const_span<scalar>& v)
{
	float max = v[0];
	uint i_max = 0;
	for (uint i = 0; i < v.size(); i++)
	{
		if (v[i] > max)
		{
			max = v[i];
			i_max = i;
		}
	}
	return i_max;
}

/*************************************************************************************************************************************/

trainer::trainer(model& m, optimizer_type& opt, const loss_function& loss) :
	_model(m), _loss(loss)
{
	for (auto& b : m.parameters())
	{
		_parameters.push_back(parameter(b, opt.for_param(b.p.size())));
	}
}

trainer::~trainer() {}

/*************************************************************************************************************************************/

void trainer::train(
	const std::vector<data>&  x_train,
	const std::vector<label>& y_train,
	const std::vector<data>&  x_test,
	const std::vector<label>& y_test,
	size_t       epochs,
	uint         batch_size
)
{
	assert(x_train.size() % batch_size == 0);
	assert(x_test.size() % batch_size == 0);
	assert(x_train.size() == y_train.size());
	assert(x_test.size() == y_test.size());

	uint input_size  = _model.input_shape().total_size();
	uint output_size = _model.output_shape().total_size();

	for (size_t ep = 0; ep < epochs; ep++)
	{
		std::cout << time_stamp << " epoch " << ep << ":" << std::endl;

		auto first = std::chrono::system_clock::now();
		auto last = first;
		uint i_count = 0;
		uint i_iters = 0;
		float training_loss = 0.0f;

		progress_printer pro(x_train.size() / batch_size);

		foreach_random_batch(_model, batch_size, x_train, y_train, [&](auto input, auto target)
		{
			pro.next();

			auto dc = device::get().scope(execution_mode::training, batch_size);

			auto x  = dc.batch_alloc(input_size);
			auto y  = dc.batch_alloc(output_size);

			dc.update(x, input);
			dc.update(y, target);

			//training
			auto r = train_batch(dc, x, y);

			training_loss += _loss.loss(dc, r.y, y);

			i_count++;
			i_iters++;
		});

		training_loss /= x_train.size();

		auto metrics = evaluate(x_test, y_test, batch_size);

		std::cout << "training loss: " << training_loss
			<< " | loss: " << metrics.loss
			<< " | accuracy: " << metrics.accuracy << std::endl;
	}
}

trainer::metrics trainer::evaluate(
	const std::vector<data>& x_test,
	const std::vector<label>& y_test,
	uint batch_size
)
{
	metrics mt;
	mt.loss = 0;
	uint correct = 0;

	uint input_size = _model.input_shape().total_size();
	uint output_size = _model.output_shape().total_size();

	std::vector<scalar> pred_buf;

	foreach_random_batch(_model, batch_size, x_test, y_test, [&](auto input, auto target)
	{
		auto dc = device::get().scope(execution_mode::execute, batch_size);

		auto t = dc.batch_alloc(output_size);
		auto x = dc.batch_alloc(input_size);

		dc.update(t, target);
		dc.update(x, input);

		auto a = _model.forward(dc, x);

		// calculate loss
		mt.loss += _loss.loss(dc, a, t);

		// direct comparison
		dc.read(a, pred_buf);
		tensor<2> h_prediction(pred_buf.data(), t.layout());
		tensor<2> h_target(const_cast<scalar*>(target.begin()), t.layout());

		for (uint b = 0; b < batch_size; b++)
		{
			if (arg_max(h_prediction[b]) == arg_max(h_target[b]))
			{
				correct++;
			}
		}
	});

	mt.loss /= x_test.size();
	mt.accuracy = (float)correct / x_test.size();
	return mt;
}

trainer::result trainer::train_batch(scope& dc, const tensor<2>& x, const tensor<2>& y)
{
	result r;
	//forward prop
	r.y = _model.forward(dc, x);
	//backward prop
	r.dy = _model.backward(dc, _loss.grad(dc, r.y, y));
	//optimize
	update_parameters();

	return r;
}

void trainer::update_parameters()
{
	for (auto& p : _parameters)
	{
		p.optimize(p.param, p.grad);
	}
}

/*************************************************************************************************************************************/

progress_printer::progress_printer(size_t count) :
	_total(count), _counter(0), _iters(0), _last(std::chrono::system_clock::now())
{}

void progress_printer::next()
{
	_counter++;
	_iters++;

	if (_counter >= _total)
	{
		stop();
	}
	else
	{
		auto t = std::chrono::system_clock::now();
		if ((t - _last) > std::chrono::seconds(1))
		{
			std::cout << "(" << _counter << "/" << _total << ") ";
			std::cout << std::fixed << std::setprecision(3) << (100 * (float)_counter / _total) << "% | ";
			std::cout << _iters << "it/s                              \r";
			_last = t;
			_iters = 0;
		}
	}
}

void progress_printer::stop()
{
	std::cout << "(" << _total << "/" << _total << ") 100%                                        " << std::endl;
}

/*************************************************************************************************************************************/

void nn::foreach_random_batch(model& m, uint batch_size, const std::vector<trainer::data>& dataset, const std::vector<trainer::label>& labels, const training_function& func)
{
	assert(dataset.size() == labels.size());
	assert((dataset.size() % batch_size) == 0);

	auto rng = new_random_engine();

	auto number_of_classes = m.output_shape().total_size();

	std::vector<size_t> indices(dataset.size());
	std::vector<scalar> x;
	std::vector<scalar> y;

	std::iota(indices.begin(), indices.end(), 0);
	std::shuffle(indices.begin(), indices.end(), rng);

	for (size_t i = 0; i < indices.size(); i += batch_size)
	{
		x.clear();
		y.clear();

		for (size_t i_batch = 0; i_batch < batch_size; i_batch++)
		{
			size_t i_sample = indices[i + i_batch];
			const auto& data = dataset[i_sample];
			uint label = labels[i_sample];

			for (scalar v : data)
				x.push_back(v);

			if (number_of_classes == 1)
			{
				y.push_back((scalar)label);
			}
			else
			{
				// one-hot encoding
				for (uint i = 0; i < number_of_classes; i++)
					y.push_back(i == label ? 1.0f : 0.0f);
			}
		}

		func(x, y);
	}
}

/*************************************************************************************************************************************/