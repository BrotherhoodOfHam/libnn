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

		gan(model& g, model& d) :
			_g(g), _d(d)
		{
			assert(tensor_shape::equals(_g.output_shape(), _d.input_shape()));
			assert(_d.output_shape()[0] == 1);
		}

		void train(const std::vector<tensor>& data, size_t epochs);

	private:

		void save_generated_images(size_t id);
	};
}
