/*
	Neural Network Model
*/

#pragma once


#include "common.h"
#include "sequence.h"

/*************************************************************************************************************************************/

namespace nn
{
	class model : public sequence
	{
		layout<2> _input_layout;
		float _learning_rate;

	public:

		model(size_t input_size, size_t max_batch_size, float learning_rate);
		~model();

		model(const model&) = delete;

		void train(
			const std::vector<buffer>& x_train,
			const std::vector<buffer>& y_train,
			const std::vector<buffer>& x_test,
			const std::vector<buffer>& y_test,
			size_t epochs
		);

		float train_batch(const buffer& x, const buffer& y);

		void train_from_gradient(const buffer& dy);

		const buffer& forward_backwards(const buffer& x, const buffer& y);

		// serialization
		bool serialize(const std::string& filename);
		bool deserialize(const std::string& filename);

	private:

		void loss_derivative(const buffer& y, const buffer& t, buffer& dy);
	};
}
