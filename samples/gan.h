/*
	GAN
*/

#pragma once

#include "nn/model.h"
#include "nn/training.h"

namespace nn
{
	class gan
	{
		pool_allocator _pool;
		context _dc;

		model& _g;
		model& _d;

	public:

		gan(model& g, model& d);
		void train(const std::vector<trainer::data>& data, uint epochs, uint batch_size);

	private:

		void save_generated_images(uint id, const tensor<2>& z_batch);
	};
}
