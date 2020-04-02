/*
	Training
*/

#pragma once

#include "model.h"

namespace nn
{
	class trainer
	{
		model&    _model;
		layout<2> _input_layout;
		layout<2> _output_layout;
		float     _learning_rate;

		std::vector<node_parameter> _parameters;

		void update_parameters();

	public:

		using data = std::vector<scalar>;
		using label = uint8_t;

		trainer(model& seq, float learning_rate);
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
}
