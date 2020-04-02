/*
	GAN
*/

#pragma once

#include "nn/model.h"

namespace nn
{
	class gan
	{
		model& _g;
		model& _d;

	public:

		gan(model& g, model& d);
		void train(const std::vector<buffer>& data, uint epochs);

	private:

		void save_generated_images(uint id);
	};
}
