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

		using data = std::vector<scalar>;
		using label = uint8_t;

		model(size_t input_size, size_t max_batch_size, float learning_rate);
		~model();

		model(const model&) = delete;

		void train(
			const std::vector<data>&  x_train,
			const std::vector<label>& y_train,
			const std::vector<data>&  x_test,
			const std::vector<label>& y_test,
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
