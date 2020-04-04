/*
	Training
*/

#pragma once

#include "model.h"
#include "optimizers.h"

namespace nn
{
	class trainer
	{
		model&    _model;
		layout<2> _input_layout;
		layout<2> _output_layout;

		struct parameter
		{
			buffer param;
			buffer grad;
			optimization_function optimize;

			parameter(const buffer& p, const buffer& g, optimization_function opt) :
				param(p), grad(g), optimize(opt)
			{}
		};

		std::vector<parameter> _parameters;

		void update_parameters();

	public:

		using data = std::vector<scalar>;
		using label = uint8_t;

		trainer(model& seq, float learning_rate = 0.01f, float momentum = 0.0f);
		~trainer();

		trainer(const trainer&) = delete;

		void train(
			const std::vector<data>& x_train,
			const std::vector<label>& y_train,
			const std::vector<data>& x_test,
			const std::vector<label>& y_test,
			size_t epochs
		);

		float train_batch(const buffer& x, const buffer& y);

		void train_from_gradient(const buffer& dy);

		const buffer& forward_backwards(const buffer& x, const buffer& y);

	private:

		void loss_derivative(const buffer& y, const buffer& t, buffer& dy);
	};

	template<typename func_type, class = std::enable_if_t<std::is_invocable_v<func_type, span<trainer::data>>>>
	void foreach_training_batch(const std::vector<trainer::data>& data, uint batch_size, const func_type& callback)
	{

	}
}
