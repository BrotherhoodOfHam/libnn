/*
	GAN
*/

#include <string>

#include "CImg.h"
#include "gan.h"
#include "nn/training.h"

using namespace nn;
using namespace cimg_library;

/*************************************************************************************************************************************/

gan::gan(model& g, model& d) :
	_g(g), _d(d)
{
	assert(node_shape::equals(_g.output_shape(), _d.input_shape()));
	assert(_d.output_shape()[1] == 1);
}

void gan::train(const std::vector<trainer::data>& data, uint epochs)
{
	tensor<2> z_input(_g.input_shape());
	tensor<2> real_input(_d.input_shape());
	tensor<2> y_d_1(_d.output_shape());  tensor_fill(y_d_1, 0.9f);
	tensor<2> y_d_0(_d.output_shape());  tensor_fill(y_d_0, 0.0f);
	tensor<2> y_g(_d.output_shape());    tensor_fill(y_g,   1.0f);
	
	std::uniform_real_distribution<double> unif(-1, 1);

	// testing batch
	tensor<2> z_batch_test(_g.input_shape());
	thread_local auto rng = new_random_engine();
	tensor_fill(z_batch_test, [&](){ return (scalar)unif(rng); });
	assert(z_batch_test.shape(0) >= 5 * 5);
	
	const uint batch_size = z_input.shape(0);
	const float alpha = 0.0002f;
	const float beta = 0.5f;

	assert(data.size() % batch_size == 0);

	trainer d_trainer(_d, adam(alpha, beta));
	trainer g_trainer(_g, adam(alpha, beta));

	std::vector<size_t> indices(data.size());
	std::iota(indices.begin(), indices.end(), 0);

	for (uint e = 0; e < epochs; e++)
	{
		thread_local auto rng = new_random_engine();

		std::cout << time_stamp << " epoch: " << e << std::endl;

		auto first = std::chrono::system_clock::now();
		auto last = first;

		std::shuffle(indices.begin(), indices.end(), rng);

		for (uint i = 0, iters = 0; i < indices.size(); i++, iters++)
		{
			auto t = std::chrono::system_clock::now();
			if ((t - last) > std::chrono::seconds(1))
			{
				std::cout << "(" << i << "/" << data.size() << ") ";
				std::cout << std::fixed << std::setprecision(3) << (100 * (float)i / data.size()) << "% | ";
				std::cout << iters << "it/s                              \r";
				last = t;
				iters = 0;
			}

			// update batch
			uint batch_index = i % batch_size;
			tensor_update(real_input[batch_index], data[indices[i]]);
			
			if (batch_index == 0 && i > 0)
			{
				//randomize z
				tensor_fill(z_input, [&]() { return (scalar)unif(rng); });

				d_trainer.train_batch(real_input.data(), y_d_1.data());
				d_trainer.train_batch(_g.forward(z_input.data()), y_d_0.data());

				//randomize z
				tensor_fill(z_input, [&]() { return (scalar)unif(rng); });

				const auto& dy = d_trainer.forward_backwards(_g.forward(z_input.data()), y_g.data());
				g_trainer.train_from_gradient(dy);
			}
		}

		std::cout << "(" << data.size() << "/" << data.size() << ") 100%                                       ";
		std::cout << std::endl;
		
		save_generated_images(e, z_batch_test);
		_g.serialize(std::string("img/model-" + std::to_string(e) + ".bin"));
	}
}

/*************************************************************************************************************************************/

void gan::save_generated_images(uint id, const tensor<2>& z_batch)
{
	const std::string filename = "img/g" + std::to_string(id) + ".bmp";

	const uint scale_factor = 16;
	const uint tile_wh = 28 * scale_factor;
	const uint border_sz = 24;
	const uint tile_count = 5;
	const uint total_wh = (tile_wh * tile_count) + (border_sz * (tile_count + 1));

	CImg<float> image(total_wh, total_wh);

	layout<2> img_layout(z_batch.shape(0), 28 * 28);
	auto g = _g.forward(z_batch.data()).as_tensor(img_layout);

	uint batch_index = 0;
	for (uint y_tile = 0; y_tile < tile_count; y_tile++)
	{
		for (uint x_tile = 0; x_tile < tile_count; x_tile++)
		{
			const uint x = (x_tile * tile_wh) + ((x_tile + 1) * border_sz);
			const uint y = (y_tile * tile_wh) + ((y_tile + 1) * border_sz);

			for (uint i_g = 0; i_g < g.shape(1); i_g++)
			{
				const uint w = i_g % 28;
				const uint h = i_g / 28;

				for (uint i_sub = 0; i_sub < (scale_factor * scale_factor); i_sub++)
				{
					const uint sub_w = i_sub % scale_factor;
					const uint sub_h = i_sub / scale_factor;

					image(
						x + (w * scale_factor) + sub_w,
						y + (h * scale_factor) + sub_h
					) = ((g[batch_index][i_g] + 1) / 2) * 255;
				}
			}

			batch_index++;
		}
	}

	image.save(filename.c_str());
	std::cout << time_stamp << " saved: " << filename << std::endl;
}

/*************************************************************************************************************************************/
