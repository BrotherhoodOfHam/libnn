/*
	GAN
*/

#pragma once

#include "nn/model.h"
#include "nn/training.h"

namespace nn
{
	/*
		GAN trainer
	*/
	class GAN
	{
		model&  _g;
		model&  _d;
		pool_allocator _constants;

	public:

		GAN(model& g, model& d);
		void train(const std::vector<trainer::data>& data, uint epochs, uint batch_size);

	private:

		void save_generated_images(uint id, const tensor<2>& z_batch);
	};
}
