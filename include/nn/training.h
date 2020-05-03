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

		std::vector<parameter> _parameters;

		void update_parameters();

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

		// training result
		struct result
		{
			batch y;
			batch dy;
		};

		// Train on a batch
		result train_batch(scope& dc, const batch& x, const batch& y);

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

	/*
		Progess pretty printer
	*/
	class progress_printer
	{
		std::chrono::time_point<std::chrono::system_clock> _last;
		size_t _counter;
		size_t _total;
		size_t _iters;

	public:

		progress_printer(size_t count);

		void next();
		void stop();
	};

	/*
		Helper function for iterating over batches of a dataset
	*/
	using training_function = std::function<void(const_span<scalar>, const_span<scalar>)>;

	void foreach_random_batch(model& m, uint batch_size, const std::vector<trainer::data>& dataset, const std::vector<trainer::label>& labels, const training_function& func);
}
