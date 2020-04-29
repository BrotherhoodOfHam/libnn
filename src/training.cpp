/*
	Neural Network Model
*/

#include <cmath>
#include <memory>
#include <iostream>
#include <random>
#include <chrono>

#include "device/kernels.h"
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

template<typename function_type, typename = if_callable<function_type, const const_span<scalar>&, const const_span<scalar>&>>
void foreach_batch(uint batch_size, uint number_of_classes, const std::vector<trainer::data>& dataset, const std::vector<trainer::label>& labels, const function_type& func)
{
	assert(dataset.size() == labels.size());
	assert((dataset.size() % batch_size) == 0);

	auto rng = new_random_engine();

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

		uint batch_count = (uint)(x_train.size() / batch_size);

		foreach_batch(batch_size, output_size, x_train, y_train, [&](const_span<scalar> inp, const_span<scalar> out)
		{
			auto t = std::chrono::system_clock::now();
			if ((t - last) > std::chrono::seconds(1))
			{
				std::cout << "(" << i_count << "/" << batch_count << ") ";
				std::cout << std::round(1000.0f * (float)i_count / batch_count) / 10 << "% | " << i_iters << "it/s                            \r";
				last = t;
				i_iters = 0;
			}

			_dc.set_batches(batch_size);
			_dc.set_mode(execution_mode::training);
			_dc.clear_allocator();

			auto x  = _dc.batch_alloc(input_size);
			auto y  = _dc.batch_alloc(output_size);

			_dc.update(x, inp);
			_dc.update(y, out);

			//training
			train_batch(x, y);

			i_count++;
			i_iters++;
		});

		std::cout << "(" << i_count << "/" << batch_count << ") 100%";
		std::cout << std::endl;

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

	_dc.set_mode(execution_mode::execute);
	_dc.set_batches(batch_size);

	std::vector<scalar> pred_buf;

	foreach_batch(batch_size, output_size, x_test, y_test, [&](const_span<scalar> input, const_span<scalar> target)
	{
		_dc.clear_allocator();

		auto t = _dc.batch_alloc(output_size);
		auto x = _dc.batch_alloc(input_size);

		_dc.update(t, target);
		_dc.update(x, input);

		auto a = _model.forward(_dc, x);

		// calculate loss
		mt.loss += _loss.loss(_dc, a, t);

		// direct comparison
		_dc.read(a, pred_buf);
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


trainer::result trainer::train_batch(const tensor<2>& x, const tensor<2>& y)
{
	result r;
	//forward prop
	r.y = _model.forward(_dc, x);
	//backward prop
	r.dy = _model.backward(_dc, _loss.grad(_dc, r.y, y));
	//optimize
	update_parameters();

	return r;
}

void trainer::train(const tensor<2>& x, const tensor<2>& y)
{
	assert(x.shape(0) == y.shape(0));

	_dc.set_batches(x.shape(0));
	_dc.set_mode(execution_mode::training);
	_dc.clear_allocator();

	train_batch(x, y);
}

/*************************************************************************************************************************************/

void trainer::update_parameters()
{
	for (auto& p : _parameters)
	{
		p.optimize(p.param, p.grad);
	}
}

/*************************************************************************************************************************************/
