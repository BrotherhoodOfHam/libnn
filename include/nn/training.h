/*
	Training
*/

#pragma once

#include "device.h"
#include "model.h"
#include "optimizers.h"
#include "losses.h"

namespace nn
{
	class trainer_routine
	{
	public:

		using data = std::vector<scalar>;
		using label = uint8_t;

		virtual void iterate(span<data> x, span<label> y)
		{

		}
	};

	class trainer : public trainer_routine
	{
		context       _dc;
		model&        _model;
		loss_function _loss;

		struct parameter
		{
			buffer param;
			buffer grad;
			opt_function optimize;

			parameter(const buffer& _param, const buffer& _grad, opt_function opt) :
				param(_param), grad(_grad), optimize(std::move(opt))
			{}
		};

		std::vector<parameter> _parameters;

		void update_parameters();

	public:

		using data = std::vector<scalar>;
		using label = uint8_t;

		trainer(model& seq, optimizer_type& opt, const loss_function& loss = categorical_cross_entropy());
		~trainer();

		trainer(const trainer&) = delete;

		void train(
			const std::vector<data>& x_train,
			const std::vector<label>& y_train,
			const std::vector<data>& x_test,
			const std::vector<label>& y_test,
			size_t epochs,
			uint batch_size
		);

		float train_batch(const tensor<2>& x, const tensor<2>& y);

		void train_gradient(const vector& dy);

		vector forward_backwards(const vector& x, const vector& y);
	};
}
