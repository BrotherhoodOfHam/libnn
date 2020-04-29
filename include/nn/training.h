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
	/*
		Model trainer class

		Provides facilities for training a model with a given optimization algorithm and loss function
	*/
	class trainer
	{
		context       _dc;
		model&        _model;
		loss_function _loss;

		struct parameter
		{
			vector param;
			vector grad;
			opt_function optimize;

			parameter(const node_parameter& p, opt_function opt) :
				param(p.p), grad(p.dp), optimize(std::move(opt))
			{}
		};

		// training result
		struct result
		{
			vector y;
			vector dy;
		};

		std::vector<parameter> _parameters;

		void update_parameters();
		result train_batch(const tensor<2>& x, const tensor<2>& y);

	public:

		using data = std::vector<scalar>;
		using label = uint8_t;

		trainer(model& seq, optimizer_type& opt, const loss_function& loss = categorical_cross_entropy());
		~trainer();

		trainer(const trainer&) = delete;

		// Train model on a dataset
		void train(
			const std::vector<data>& x_train,
			const std::vector<label>& y_train,
			const std::vector<data>& x_test,
			const std::vector<label>& y_test,
			size_t epochs,
			uint batch_size
		);

		// Train on a batch
		void train(const tensor<2>& x, const tensor<2>& y);

		// Evaluation metrics
		struct metrics
		{
			float loss;
			float accuracy;
		};

		// Evaluate the model performance
		metrics evaluate(
			const std::vector<data>& x_test,
			const std::vector<label>& y_test,
			uint batch_size = 1
		);
	};
}
