/*
	GAN
*/

#pragma once

#include "nn/model.h"

namespace nn
{
	class gan
	{
		model* _g;
		model* _d;

	public:

		gan(model* g, model* d) :
			_g(g), _d(d)
		{
			tensor::check(_g->output_size(), _d->input_size());
			assert(_d->output_size()[0] == 1);
		}

		void train(const std::vector<tensor>& data, size_t epochs);

	private:

		void save_generated_images(size_t id);
	};
}
