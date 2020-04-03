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
		model& _g;
		model& _d;

	public:

		gan(model& g, model& d);
		void train(const std::vector<trainer::data>& data, uint epochs);

	private:

		void save_generated_images(uint id, const tensor<2>& z_batch);
	};
}
